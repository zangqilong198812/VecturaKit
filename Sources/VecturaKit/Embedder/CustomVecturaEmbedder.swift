import Foundation

/// VecturaEmbedder that forwards to SwiftEmbedder while allowing custom download endpoints.
@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public actor CustomVecturaEmbedder {

  private let embedder: SwiftEmbedder

  /// Initializes the embedder with a remote model ID and custom Hugging Face base URL.
  ///
  /// - Parameters:
  ///   - modelId: Identifier of the model to download.
  ///   - type: Optional hint for the model architecture.
  ///   - downloadBase: Base URL that replaces the default Hugging Face endpoint.
  ///   - useBackgroundSession: Whether downloads should use a background URLSession.
  public init(
    modelId: String,
    type: VecturaModelSource.ModelType? = nil,
    downloadBase: URL,
    useBackgroundSession: Bool = false
  ) {
    self.embedder = SwiftEmbedder(
      modelSource: .id(modelId, type: type),
      loadingOptions: .init(
        downloadBase: downloadBase,
        useBackgroundSession: useBackgroundSession
      )
    )
  }

  /// Initializes the embedder with an existing model source and custom download options.
  ///
  /// - Parameters:
  ///   - modelSource: Source describing either a remote ID or local folder.
  ///   - downloadBase: Optional base URL override for remote downloads.
  ///   - useBackgroundSession: Whether downloads should use a background URLSession.
  public init(
    modelSource: VecturaModelSource,
    downloadBase: URL? = nil,
    useBackgroundSession: Bool = false
  ) {
    self.embedder = SwiftEmbedder(
      modelSource: modelSource,
      loadingOptions: .init(
        downloadBase: downloadBase,
        useBackgroundSession: useBackgroundSession
      )
    )
  }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension CustomVecturaEmbedder: VecturaEmbedder {

  public var dimension: Int {
    get async throws {
      try await embedder.dimension
    }
  }

  public func embed(texts: [String]) async throws -> [[Float]] {
    try await embedder.embed(texts: texts)
  }

  public func embed(text: String) async throws -> [Float] {
    try await embedder.embed(text: text)
  }
}
