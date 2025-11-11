'use client'

export interface TextArtifactProps {
  content: string
  metadata?: Record<string, any>
}

export function TextArtifact({ content, metadata }: TextArtifactProps) {
  return (
    <div className="space-y-4">
      {metadata && (
        <div className="text-sm text-gray-400 space-y-1">
          {metadata.similarity && (
            <div>Similarity: {(metadata.similarity * 100).toFixed(1)}%</div>
          )}
          {metadata.document_id && (
            <div>Document ID: {metadata.document_id}</div>
          )}
        </div>
      )}
      
      <div className="prose prose-invert prose-sm max-w-none">
        <div className="text-gray-300 whitespace-pre-wrap leading-relaxed">
          {content}
        </div>
      </div>
    </div>
  )
}

