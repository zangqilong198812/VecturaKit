import CoreML
import Darwin
import Embeddings
import Foundation

/// An embedder implementation using swift-embeddings library (Bert, Model2Vec, and StaticEmbeddings models).
@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public actor SwiftEmbedder {

  /// Configuration for how models are fetched from Hugging Face or custom hosts.
  public struct ModelLoadingOptions: Sendable {
    public let downloadBase: URL?
    public let useBackgroundSession: Bool
    public let endpoint: URL?

    public init(
      downloadBase: URL? = nil,
      useBackgroundSession: Bool = false,
      endpoint: URL? = nil
    ) {
      self.downloadBase = downloadBase
      self.useBackgroundSession = useBackgroundSession
      self.endpoint = endpoint
    }
  }

  private let modelSource: VecturaModelSource
  private let loadingOptions: ModelLoadingOptions
  private var bertModel: Bert.ModelBundle?
  private var model2vecModel: Model2Vec.ModelBundle?
  private var staticEmbeddingsModel: StaticEmbeddings.ModelBundle?
  private var cachedDimension: Int?

  /// Initializes a SwiftEmbedder with the specified model source.
  ///
  /// - Parameters:
  ///   - modelSource: The source from which to load the embedding model.
  ///   - loadingOptions: Extra options for downloading remote models.
  public init(
    modelSource: VecturaModelSource = .default,
    loadingOptions: ModelLoadingOptions = .init()
  ) {
    self.modelSource = modelSource
    self.loadingOptions = loadingOptions
  }
}

// MARK: - VecturaEmbedder

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension SwiftEmbedder: VecturaEmbedder {

  /// The dimensionality of the embedding vectors produced by this embedder.
  ///
  /// This value is cached after first detection to avoid repeated computation.
  /// - Throws: An error if the dimension cannot be determined.
  public var dimension: Int {
    get async throws {
      if let cached = cachedDimension {
        return cached
      }

      // Ensure model is loaded
      try await ensureModelLoaded()

      let dim: Int
      if let model2vec = model2vecModel {
        // Note: 'dimienstion' is a typo in the upstream swift-embeddings library
        // See: swift-embeddings/Sources/Embeddings/Model2Vec/Model2VecModel.swift
        dim = model2vec.model.dimienstion
      } else if let staticEmbeddings = staticEmbeddingsModel {
        dim = staticEmbeddings.model.dimension
      } else if let bert = bertModel {
        // For BERT, we need to get dimension from a test encoding
        let testEmbedding = try bert.encode("test")
        guard let lastDim = testEmbedding.shape.last else {
          throw VecturaError.invalidInput(
            "Could not determine BERT model dimension from shape \(testEmbedding.shape)"
          )
        }
        dim = lastDim
      } else {
        throw VecturaError.invalidInput("No model loaded to detect dimension")
      }

      cachedDimension = dim
      return dim
    }
  }

  /// Generates embeddings for multiple texts in batch.
  ///
  /// - Parameter texts: The text strings to embed.
  /// - Returns: An array of embedding vectors, one for each input text.
  /// - Throws: An error if embedding generation fails.
  public func embed(texts: [String]) async throws -> [[Float]] {
    try await ensureModelLoaded()

    let embeddingsTensor: MLTensor
    if let model2vec = model2vecModel {
      embeddingsTensor = try model2vec.batchEncode(texts)
    } else if let staticEmbeddings = staticEmbeddingsModel {
      embeddingsTensor = try staticEmbeddings.batchEncode(texts, normalize: true)
    } else if let bert = bertModel {
      embeddingsTensor = try bert.batchEncode(texts)
    } else {
      throw VecturaError.invalidInput("Failed to load model: \(modelSource)")
    }

    let shape = embeddingsTensor.shape
    guard shape.count == 2, let dimension = shape.last else {
      throw VecturaError.invalidInput("Expected shape [N, D], got \(shape)")
    }
    let embeddingShapedArray = await embeddingsTensor.cast(to: Float.self).shapedArray(of: Float.self)
    let allScalars = embeddingShapedArray.scalars

    return stride(from: 0, to: allScalars.count, by: dimension).map {
      Array(allScalars[$0..<($0 + dimension)])
    }
  }

  /// Generates an embedding for a single text.
  ///
  /// - Parameter text: The text string to embed.
  /// - Returns: The embedding vector for the input text.
  /// - Throws: An error if embedding generation fails.
  public func embed(text: String) async throws -> [Float] {
    try await ensureModelLoaded()

    let embeddingTensor: MLTensor
    if let model2vec = model2vecModel {
      embeddingTensor = try model2vec.encode(text)
    } else if let staticEmbeddings = staticEmbeddingsModel {
      embeddingTensor = try staticEmbeddings.encode(text, normalize: true)
    } else if let bert = bertModel {
      embeddingTensor = try bert.encode(text)
    } else {
      throw VecturaError.invalidInput("Failed to load model: \(modelSource)")
    }

    let embeddingShapedArray = await embeddingTensor.cast(to: Float.self).shapedArray(of: Float.self)
    return embeddingShapedArray.scalars
  }

  private func ensureModelLoaded() async throws {
    guard bertModel == nil && model2vecModel == nil && staticEmbeddingsModel == nil else {
      return
    }

    if isModel2VecModel(modelSource) {
      model2vecModel = try await Model2Vec.loadModelBundle(
        from: modelSource,
        loadingOptions: loadingOptions
      )
    } else if isStaticEmbeddingsModel(modelSource) {
      staticEmbeddingsModel = try await StaticEmbeddings.loadModelBundle(
        from: modelSource,
        loadingOptions: loadingOptions
      )
    } else {
      bertModel = try await Bert.loadModelBundle(
        from: modelSource,
        loadingOptions: loadingOptions
      )
    }
  }

  /// Determines if a model source refers to a Model2Vec model.
  ///
  /// First checks for an explicit model type, then falls back to string-based heuristics.
  /// The string matching covers known Model2Vec model families including minishlab, potion, and M2V variants.
  ///
  /// - Note: For best reliability, explicitly specify the model type when creating VecturaModelSource.
  /// - Parameter source: The model source to check.
  /// - Returns: `true` if the source appears to be a Model2Vec model, `false` otherwise.
  private func isModel2VecModel(_ source: VecturaModelSource) -> Bool {
    // Check explicit type first
    switch source {
    case .id(_, let type), .folder(_, let type):
      if let type = type {
        return type == .model2vec
      }
    }

    // Fall back to string matching
    let modelId = source.description
    return modelId.contains("minishlab") ||
         modelId.contains("potion") ||
         modelId.contains("model2vec") ||
         modelId.contains("M2V")
  }

  /// Determines if a model source refers to a StaticEmbeddings model based on string matching.
  ///
  /// This uses string-based heuristics to identify StaticEmbeddings models from sentence-transformers.
  /// The check covers known StaticEmbeddings model families including static-retrieval and static-similarity.
  ///
  /// - Note: This approach may need updates if new StaticEmbeddings naming schemes are introduced.
  /// - Parameter source: The model source to check.
  /// - Returns: `true` if the source appears to be a StaticEmbeddings model, `false` otherwise.
  private func isStaticEmbeddingsModel(_ source: VecturaModelSource) -> Bool {
    let modelId = source.description
    return modelId.contains("static-retrieval") ||
         modelId.contains("static-similarity") ||
         modelId.contains("static-embed")
  }
}

// MARK: - Model Loading Extensions

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Bert {

  static func loadModelBundle(
    from source: VecturaModelSource,
    loadingOptions: SwiftEmbedder.ModelLoadingOptions,
    loadConfig: LoadConfig = .init()
  ) async throws -> Bert.ModelBundle {
    switch source {
    case .id(let modelId, _):
      if let cachedURL = cachedModelDirectoryIfComplete(
        for: source,
        downloadBase: loadingOptions.downloadBase,
        loadConfig: loadConfig
      ) {
        do {
          return try await loadModelBundle(from: cachedURL, loadConfig: loadConfig)
        } catch {
          try? FileManager.default.removeItem(at: cachedURL)
        }
      }
      return try await withHubEndpoint(loadingOptions.endpoint) {
        try await loadModelBundle(
          from: modelId,
          downloadBase: loadingOptions.downloadBase,
          useBackgroundSession: loadingOptions.useBackgroundSession,
          loadConfig: loadConfig
        )
      }
    case .folder(let url, _):
      return try await loadModelBundle(from: url, loadConfig: loadConfig)
    }
  }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Model2Vec {

  static func loadModelBundle(
    from source: VecturaModelSource,
    loadingOptions: SwiftEmbedder.ModelLoadingOptions,
    loadConfig: LoadConfig = .init()
  ) async throws -> Model2Vec.ModelBundle {
    switch source {
    case .id(let modelId, _):
      if let cachedURL = cachedModelDirectoryIfComplete(
        for: source,
        downloadBase: loadingOptions.downloadBase,
        loadConfig: loadConfig
      ) {
        do {
          return try await loadModelBundle(from: cachedURL, loadConfig: loadConfig)
        } catch {
          try? FileManager.default.removeItem(at: cachedURL)
        }
      }
      return try await withHubEndpoint(loadingOptions.endpoint) {
        try await loadModelBundle(
          from: modelId,
          downloadBase: loadingOptions.downloadBase,
          useBackgroundSession: loadingOptions.useBackgroundSession,
          loadConfig: loadConfig
        )
      }
    case .folder(let url, _):
      return try await loadModelBundle(from: url, loadConfig: loadConfig)
    }
  }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension StaticEmbeddings {

  static func loadModelBundle(
    from source: VecturaModelSource,
    loadingOptions: SwiftEmbedder.ModelLoadingOptions,
    loadConfig: LoadConfig = .staticEmbeddings
  ) async throws -> StaticEmbeddings.ModelBundle {
    switch source {
    case .id(let modelId, _):
      if let cachedURL = cachedModelDirectoryIfComplete(
        for: source,
        downloadBase: loadingOptions.downloadBase,
        loadConfig: loadConfig
      ) {
        do {
          return try await loadModelBundle(from: cachedURL, loadConfig: loadConfig)
        } catch {
          try? FileManager.default.removeItem(at: cachedURL)
        }
      }
      return try await withHubEndpoint(loadingOptions.endpoint) {
        try await loadModelBundle(
          from: modelId,
          downloadBase: loadingOptions.downloadBase,
          useBackgroundSession: loadingOptions.useBackgroundSession,
          loadConfig: loadConfig
        )
      }
    case .folder(let url, _):
      return try await loadModelBundle(from: url, loadConfig: loadConfig)
    }
  }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
func cachedModelDirectoryIfComplete(
  for source: VecturaModelSource,
  downloadBase: URL?,
  loadConfig: LoadConfig
) -> URL? {
  guard case .id = source else {
    return nil
  }

  guard let baseDirectory = resolvedDownloadBase(downloadBase) else {
    return nil
  }

  let modelDirectory = baseDirectory
    .appending(path: "models", directoryHint: .isDirectory)
    .appending(path: source.description, directoryHint: .isDirectory)

  guard directoryContainsRequiredFiles(modelDirectory, loadConfig: loadConfig) else {
    return nil
  }

  return modelDirectory
}

private func resolvedDownloadBase(_ override: URL?) -> URL? {
  if let override {
    return override
  }

  guard let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
    return nil
  }

  return documentsURL.appending(path: "huggingface", directoryHint: .isDirectory)
}

private func directoryContainsRequiredFiles(
  _ directory: URL,
  loadConfig: LoadConfig
) -> Bool {
  let fileManager = FileManager.default
  let configURL = directory.appending(path: loadConfig.modelConfig.configFileName, directoryHint: .notDirectory)
  let weightsURL = directory.appending(path: loadConfig.modelConfig.weightsFileName, directoryHint: .notDirectory)

  guard fileManager.fileExists(atPath: configURL.path(percentEncoded: false)),
        fileManager.fileExists(atPath: weightsURL.path(percentEncoded: false)) else {
    return false
  }

  guard let tokenizerConfig = loadConfig.tokenizerConfig else {
    return true
  }

  return tokenizerFilesExist(tokenizerConfig.data, at: directory) &&
    tokenizerFilesExist(tokenizerConfig.config, at: directory)
}

private func tokenizerFilesExist(
  _ config: TokenizerConfigType,
  at directory: URL
) -> Bool {
  switch config {
  case .filePath(let path):
    let fileURL = directory.appending(path: path, directoryHint: .notDirectory)
    return FileManager.default.fileExists(atPath: fileURL.path(percentEncoded: false))
  case .data:
    return true
  }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
private func withHubEndpoint<T>(
  _ endpoint: URL?,
  operation: () async throws -> T
) async throws -> T {
  guard let endpoint else {
    return try await operation()
  }

  let previousValue = getenv("HF_ENDPOINT").flatMap { String(validatingCString: $0) }

  setenv("HF_ENDPOINT", endpoint.absoluteString, 1)
  defer {
    if let previousValue {
      setenv("HF_ENDPOINT", previousValue, 1)
    } else {
      unsetenv("HF_ENDPOINT")
    }
  }

  return try await operation()
}
