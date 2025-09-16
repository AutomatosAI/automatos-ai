'use client'

export function SimpleDocs() {
  return (
    <div className="p-6 space-y-6">
      <h1 className="text-3xl font-bold">Document Management</h1>
      
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded shadow border">
          <div className="text-2xl font-bold">6</div>
          <div className="text-sm text-gray-600">Total Documents</div>
        </div>
        <div className="bg-white p-4 rounded shadow border">
          <div className="text-2xl font-bold">0</div>
          <div className="text-sm text-gray-600">Processed</div>
        </div>
        <div className="bg-white p-4 rounded shadow border">
          <div className="text-2xl font-bold">15.7 MB</div>
          <div className="text-sm text-gray-600">Storage Used</div>
        </div>
        <div className="bg-white p-4 rounded shadow border">
          <div className="text-2xl font-bold">0</div>
          <div className="text-sm text-gray-600">Vector Chunks</div>
        </div>
      </div>
      
      <div className="bg-white p-6 rounded shadow border">
        <h2 className="text-xl font-semibold mb-4">Documents</h2>
        <p className="text-gray-600">Your documents page is now loading. Upload and management features are being restored.</p>
      </div>
    </div>
  )
}
