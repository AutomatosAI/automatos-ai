
"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  Upload, 
  FileText, 
  Trash2, 
  Search, 
  Filter,
  Download,
  Eye,
  AlertCircle,
  CheckCircle,
  Clock,
  XCircle
} from 'lucide-react';
import { useDropzone } from 'react-dropzone';

interface Document {
  id: number;
  filename: string;
  file_type: string;
  file_size: number;
  upload_date: string;
  processed_date?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  chunk_count: number;
  tags: string[];
  description: string;
  created_by: string;
}

interface DocumentStats {
  total_documents: number;
  status_distribution: Record<string, number>;
  total_chunks: number;
  file_type_distribution: Record<string, number>;
}

const DocumentManagementPage = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [stats, setStats] = useState<DocumentStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [uploadForm, setUploadForm] = useState({
    tags: '',
    description: '',
    created_by: 'admin'
  });

  // Fetch documents
  const fetchDocuments = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (statusFilter !== 'all') params.append('status', statusFilter);
      
      const response = await fetch(`http://localhost:8001/api/admin/documents?${params}`);
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (error) {
      console.error('Error fetching documents:', error);
    } finally {
      setLoading(false);
    }
  };

  // Fetch stats
  const fetchStats = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/admin/stats');
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  useEffect(() => {
    fetchDocuments();
    fetchStats();
  }, [statusFilter]);

  // File upload with dropzone
  const onDrop = async (acceptedFiles: File[]) => {
    for (const file of acceptedFiles) {
      await uploadFile(file);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/markdown': ['.md'],
      'text/plain': ['.txt'],
      'text/x-python': ['.py'],
      'application/json': ['.json']
    },
    multiple: true
  });

  const uploadFile = async (file: File) => {
    try {
      setUploadProgress(0);
      const formData = new FormData();
      formData.append('file', file);
      formData.append('tags', uploadForm.tags);
      formData.append('description', uploadForm.description);
      formData.append('created_by', uploadForm.created_by);

      const response = await fetch('http://localhost:8001/api/admin/documents/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        setUploadProgress(100);
        await fetchDocuments();
        await fetchStats();
        
        // Reset form
        setUploadForm({
          tags: '',
          description: '',
          created_by: 'admin'
        });
      } else {
        throw new Error('Upload failed');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  const deleteDocument = async (id: number) => {
    if (!confirm('Are you sure you want to delete this document?')) return;
    
    try {
      const response = await fetch(`http://localhost:8001/api/admin/documents/${id}`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        await fetchDocuments();
        await fetchStats();
      }
    } catch (error) {
      console.error('Error deleting document:', error);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'processing':
        return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <AlertCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'processing':
        return 'bg-yellow-100 text-yellow-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const filteredDocuments = documents.filter(doc =>
    doc.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
    doc.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
    doc.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Document Management</h1>
          <p className="text-muted-foreground">
            Manage documents for the context engineering system
          </p>
        </div>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Documents</CardTitle>
              <FileText className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_documents}</div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Chunks</CardTitle>
              <FileText className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_chunks}</div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Completed</CardTitle>
              <CheckCircle className="h-4 w-4 text-green-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.status_distribution.completed || 0}</div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Processing</CardTitle>
              <Clock className="h-4 w-4 text-yellow-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.status_distribution.processing || 0}</div>
            </CardContent>
          </Card>
        </div>
      )}

      <Tabs defaultValue="documents" className="space-y-4">
        <TabsList>
          <TabsTrigger value="documents">Documents</TabsTrigger>
          <TabsTrigger value="upload">Upload</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="documents" className="space-y-4">
          {/* Search and Filter */}
          <div className="flex gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search documents..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-8"
                />
              </div>
            </div>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-3 py-2 border rounded-md"
            >
              <option value="all">All Status</option>
              <option value="pending">Pending</option>
              <option value="processing">Processing</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
            </select>
          </div>

          {/* Documents List */}
          <div className="grid gap-4">
            {loading ? (
              <div className="text-center py-8">Loading documents...</div>
            ) : filteredDocuments.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No documents found
              </div>
            ) : (
              filteredDocuments.map((doc) => (
                <Card key={doc.id}>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <h3 className="font-semibold">{doc.filename}</h3>
                          <Badge className={getStatusColor(doc.status)}>
                            {getStatusIcon(doc.status)}
                            <span className="ml-1">{doc.status}</span>
                          </Badge>
                          <Badge variant="outline">{doc.file_type}</Badge>
                        </div>
                        
                        <p className="text-sm text-muted-foreground mb-2">
                          {doc.description || 'No description'}
                        </p>
                        
                        <div className="flex items-center gap-4 text-sm text-muted-foreground">
                          <span>Size: {formatFileSize(doc.file_size)}</span>
                          <span>Chunks: {doc.chunk_count}</span>
                          <span>Uploaded: {new Date(doc.upload_date).toLocaleDateString()}</span>
                          {doc.processed_date && (
                            <span>Processed: {new Date(doc.processed_date).toLocaleDateString()}</span>
                          )}
                        </div>
                        
                        {doc.tags.length > 0 && (
                          <div className="flex gap-1 mt-2">
                            {doc.tags.map((tag, index) => (
                              <Badge key={index} variant="secondary" className="text-xs">
                                {tag}
                              </Badge>
                            ))}
                          </div>
                        )}
                      </div>
                      
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setSelectedDocument(doc)}
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => deleteDocument(doc.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </TabsContent>

        <TabsContent value="upload" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Upload Documents</CardTitle>
              <CardDescription>
                Upload PDF, DOCX, Markdown, Text, Python, or JSON files for processing
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Upload Form */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="tags">Tags (comma-separated)</Label>
                  <Input
                    id="tags"
                    value={uploadForm.tags}
                    onChange={(e) => setUploadForm({...uploadForm, tags: e.target.value})}
                    placeholder="business, process, documentation"
                  />
                </div>
                <div>
                  <Label htmlFor="created_by">Created By</Label>
                  <Input
                    id="created_by"
                    value={uploadForm.created_by}
                    onChange={(e) => setUploadForm({...uploadForm, created_by: e.target.value})}
                  />
                </div>
              </div>
              
              <div>
                <Label htmlFor="description">Description</Label>
                <Textarea
                  id="description"
                  value={uploadForm.description}
                  onChange={(e) => setUploadForm({...uploadForm, description: e.target.value})}
                  placeholder="Brief description of the document content"
                />
              </div>

              {/* Dropzone */}
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                {isDragActive ? (
                  <p>Drop the files here...</p>
                ) : (
                  <div>
                    <p className="text-lg font-medium">Drag & drop files here</p>
                    <p className="text-muted-foreground">or click to select files</p>
                    <p className="text-sm text-muted-foreground mt-2">
                      Supported: PDF, DOCX, MD, TXT, PY, JSON
                    </p>
                  </div>
                )}
              </div>

              {uploadProgress > 0 && uploadProgress < 100 && (
                <div>
                  <Label>Upload Progress</Label>
                  <Progress value={uploadProgress} className="mt-2" />
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* File Type Distribution */}
            {stats && (
              <Card>
                <CardHeader>
                  <CardTitle>File Type Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {Object.entries(stats.file_type_distribution).map(([type, count]) => (
                      <div key={type} className="flex justify-between items-center">
                        <span className="capitalize">{type}</span>
                        <Badge variant="secondary">{count}</Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Status Distribution */}
            {stats && (
              <Card>
                <CardHeader>
                  <CardTitle>Status Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {Object.entries(stats.status_distribution).map(([status, count]) => (
                      <div key={status} className="flex justify-between items-center">
                        <div className="flex items-center gap-2">
                          {getStatusIcon(status)}
                          <span className="capitalize">{status}</span>
                        </div>
                        <Badge variant="secondary">{count}</Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>
      </Tabs>

      {/* Document Preview Modal */}
      {selectedDocument && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <Card className="max-w-2xl w-full max-h-[80vh] overflow-auto">
            <CardHeader>
              <div className="flex justify-between items-start">
                <div>
                  <CardTitle>{selectedDocument.filename}</CardTitle>
                  <CardDescription>
                    {selectedDocument.description || 'No description'}
                  </CardDescription>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setSelectedDocument(null)}
                >
                  ×
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <strong>File Type:</strong> {selectedDocument.file_type}
                  </div>
                  <div>
                    <strong>Size:</strong> {formatFileSize(selectedDocument.file_size)}
                  </div>
                  <div>
                    <strong>Status:</strong> 
                    <Badge className={`ml-2 ${getStatusColor(selectedDocument.status)}`}>
                      {selectedDocument.status}
                    </Badge>
                  </div>
                  <div>
                    <strong>Chunks:</strong> {selectedDocument.chunk_count}
                  </div>
                  <div>
                    <strong>Uploaded:</strong> {new Date(selectedDocument.upload_date).toLocaleString()}
                  </div>
                  {selectedDocument.processed_date && (
                    <div>
                      <strong>Processed:</strong> {new Date(selectedDocument.processed_date).toLocaleString()}
                    </div>
                  )}
                </div>
                
                {selectedDocument.tags.length > 0 && (
                  <div>
                    <strong>Tags:</strong>
                    <div className="flex gap-1 mt-1">
                      {selectedDocument.tags.map((tag, index) => (
                        <Badge key={index} variant="secondary">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};

export default DocumentManagementPage;
