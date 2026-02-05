'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import {
  Search,
  Upload,
  FolderOpen,
  PlayCircle,
  Music,
  FileText,
  Grid3x3,
  Download,
  MoreVertical,
  Eye,
  Trash2,
  ChevronRight,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'

interface FileCategory {
  id: string
  name: string
  icon: React.ComponentType<any>
  fileCount: number
  size: string
  usagePercent: number
  color: string
  bgColor: string
}

interface Folder {
  id: string
  name: string
  fileCount: number
  size: string
}

interface RecentFile {
  id: string
  name: string
  category: string
  size: string
  dateModified: string
  icon: React.ComponentType<any>
}

interface StorageBreakdown {
  category: string
  percentage: number
  color: string
}

interface ModernFileManagerProps {
  categories?: FileCategory[]
  folders?: Folder[]
  recentFiles?: RecentFile[]
  storageBreakdown?: StorageBreakdown[]
  totalStorage?: string
  usedStorage?: string
  freeStorage?: string
  onUpload?: () => void
  onFolderClick?: (folderId: string) => void
  onFileAction?: (fileId: string, action: 'view' | 'delete') => void
}

const defaultCategories: FileCategory[] = [
  {
    id: 'images',
    name: 'Images',
    icon: FolderOpen,
    fileCount: 245,
    size: '26.40 GB',
    usagePercent: 17,
    color: 'text-emerald-600',
    bgColor: 'bg-emerald-50',
  },
  {
    id: 'videos',
    name: 'Videos',
    icon: PlayCircle,
    fileCount: 245,
    size: '26.40 GB',
    usagePercent: 22,
    color: 'text-pink-600',
    bgColor: 'bg-pink-50',
  },
  {
    id: 'audios',
    name: 'Audios',
    icon: Music,
    fileCount: 830,
    size: '18.90 GB',
    usagePercent: 23,
    color: 'text-blue-600',
    bgColor: 'bg-blue-50',
  },
  {
    id: 'apps',
    name: 'Apps',
    icon: Grid3x3,
    fileCount: 1200,
    size: '85.30 GB',
    usagePercent: 65,
    color: 'text-orange-600',
    bgColor: 'bg-orange-50',
  },
  {
    id: 'documents',
    name: 'Documents',
    icon: FileText,
    fileCount: 78,
    size: '5.40 GB',
    usagePercent: 10,
    color: 'text-amber-600',
    bgColor: 'bg-amber-50',
  },
  {
    id: 'downloads',
    name: 'Downloads',
    icon: Download,
    fileCount: 245,
    size: '26.40 GB',
    usagePercent: 16,
    color: 'text-indigo-600',
    bgColor: 'bg-indigo-50',
  },
]

const defaultFolders: Folder[] = [
  { id: '1', name: 'Images', fileCount: 345, size: '26.40 GB' },
  { id: '2', name: 'Documents', fileCount: 130, size: '26.40 GB' },
  { id: '3', name: 'Apps', fileCount: 130, size: '26.40 GB' },
  { id: '4', name: 'Downloads', fileCount: 345, size: '26.40 GB' },
]

const defaultRecentFiles: RecentFile[] = [
  {
    id: '1',
    name: 'Video_947954.mp4',
    category: 'Video',
    size: '89 MB',
    dateModified: '12 Jan, 2027',
    icon: PlayCircle,
  },
  {
    id: '2',
    name: 'Travel.jpg',
    category: 'Image',
    size: '5.4 MB',
    dateModified: '10 Feb, 2027',
    icon: FolderOpen,
  },
  {
    id: '3',
    name: 'Document.pdf',
    category: 'Document',
    size: '1.2 MB',
    dateModified: '8 Mar, 2027',
    icon: FileText,
  },
  {
    id: '4',
    name: 'Video_947954_028.mp4',
    category: 'Video',
    size: '489 MB',
    dateModified: '29 Apr, 2027',
    icon: PlayCircle,
  },
  {
    id: '5',
    name: 'Mountain.png',
    category: 'Image',
    size: '5.4 MB',
    dateModified: '10 Feb, 2027',
    icon: FolderOpen,
  },
]

const defaultStorageBreakdown: StorageBreakdown[] = [
  { category: 'Downloads', percentage: 30, color: '#818CF8' },
  { category: 'Apps', percentage: 25, color: '#FB923C' },
  { category: 'Documents', percentage: 20, color: '#FCD34D' },
  { category: 'Media', percentage: 25, color: '#34D399' },
]

export function ModernFileManager({
  categories = defaultCategories,
  folders = defaultFolders,
  recentFiles = defaultRecentFiles,
  storageBreakdown = defaultStorageBreakdown,
  totalStorage = '160 GB',
  usedStorage = '160',
  freeStorage = '585 GB',
  onUpload,
  onFolderClick,
  onFileAction,
}: ModernFileManagerProps) {
  const [searchTerm, setSearchTerm] = useState('')

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-[1440px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <span className="text-gray-900 font-medium cursor-pointer hover:text-gray-600 transition-colors">
                Home
              </span>
              <ChevronRight className="w-4 h-4" />
              <span className="text-gray-900 font-semibold">File Manager</span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-[1440px] mx-auto px-6 py-8">
        {/* Page Title and Actions */}
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-2xl font-bold text-gray-900">Automatos Storage</h1>
          <div className="flex items-center gap-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <Input
                placeholder="Search..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 w-[280px] bg-white border-gray-200 focus:border-blue-500 focus:ring-blue-500"
              />
            </div>
            <Button
              onClick={onUpload}
              className="bg-blue-600 hover:bg-blue-700 text-white transition-colors"
            >
              <Upload className="w-4 h-4 mr-2" />
              Upload File
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">

            {/* All Folders Section */}
            <Card className="bg-white border border-gray-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-lg font-bold text-gray-900">All Folders</h2>
                  <button className="flex items-center gap-1 text-sm text-blue-600 hover:text-blue-700 transition-colors font-medium">
                    View All
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {folders.map((folder) => (
                    <div
                      key={folder.id}
                      onClick={() => onFolderClick?.(folder.id)}
                      className="flex items-center justify-between p-4 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors cursor-pointer group"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-12 h-12 bg-amber-100 rounded-lg flex items-center justify-center">
                          <FolderOpen className="w-6 h-6 text-amber-600" />
                        </div>
                        <div>
                          <p className="text-sm font-semibold text-gray-900">{folder.name}</p>
                          <p className="text-xs text-gray-500">{folder.fileCount} Files</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-sm font-medium text-gray-700">{folder.size}</span>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <button className="p-1 hover:bg-gray-200 rounded transition-colors">
                              <MoreVertical className="w-4 h-4 text-gray-600" />
                            </button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem>
                              <Eye className="w-4 h-4 mr-2" />
                              View
                            </DropdownMenuItem>
                            <DropdownMenuItem className="text-red-600">
                              <Trash2 className="w-4 h-4 mr-2" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Recent Files Table */}
            <Card className="bg-white border border-gray-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-lg font-bold text-gray-900">Recent Files</h2>
                  <button className="flex items-center gap-1 text-sm text-blue-600 hover:text-blue-700 transition-colors font-medium">
                    View All
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-3 px-2 text-xs font-semibold text-gray-600 uppercase tracking-wider">
                          File Name
                        </th>
                        <th className="text-left py-3 px-2 text-xs font-semibold text-gray-600 uppercase tracking-wider">
                          Category
                        </th>
                        <th className="text-left py-3 px-2 text-xs font-semibold text-gray-600 uppercase tracking-wider">
                          Size
                        </th>
                        <th className="text-left py-3 px-2 text-xs font-semibold text-gray-600 uppercase tracking-wider">
                          Date Modified
                        </th>
                        <th className="text-right py-3 px-2 text-xs font-semibold text-gray-600 uppercase tracking-wider">
                          Action
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {recentFiles.map((file) => {
                        const Icon = file.icon
                        return (
                          <tr
                            key={file.id}
                            className="border-b border-gray-100 hover:bg-gray-50 transition-colors"
                          >
                            <td className="py-4 px-2">
                              <div className="flex items-center gap-3">
                                <Icon className="w-5 h-5 text-gray-600" />
                                <span className="text-sm font-medium text-gray-900">{file.name}</span>
                              </div>
                            </td>
                            <td className="py-4 px-2">
                              <span className="text-sm text-gray-600">{file.category}</span>
                            </td>
                            <td className="py-4 px-2">
                              <span className="text-sm text-gray-600">{file.size}</span>
                            </td>
                            <td className="py-4 px-2">
                              <span className="text-sm text-gray-600">{file.dateModified}</span>
                            </td>
                            <td className="py-4 px-2">
                              <div className="flex items-center justify-end gap-2">
                                <button
                                  onClick={() => onFileAction?.(file.id, 'view')}
                                  className="p-1.5 hover:bg-gray-200 rounded transition-colors"
                                >
                                  <Eye className="w-4 h-4 text-gray-600" />
                                </button>
                                <button
                                  onClick={() => onFileAction?.(file.id, 'delete')}
                                  className="p-1.5 hover:bg-gray-200 rounded transition-colors"
                                >
                                  <Trash2 className="w-4 h-4 text-gray-600" />
                                </button>
                              </div>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Storage Details Sidebar */}
          <div className="lg:col-span-1">
            <Card className="bg-white border border-gray-200 sticky top-6">
              <CardContent className="p-6">
                <h2 className="text-lg font-bold text-gray-900 mb-4">Storage Details</h2>
                <p className="text-sm text-gray-600 mb-6">{freeStorage} Free space left</p>

                {/* Donut Chart */}
                <div className="relative w-full aspect-square max-w-[240px] mx-auto mb-6">
                  <svg viewBox="0 0 100 100" className="transform -rotate-90">
                    {(() => {
                      let cumulativePercent = 0
                      return storageBreakdown.map((item, index) => {
                        const strokeDasharray = `${item.percentage} ${100 - item.percentage}`
                        const strokeDashoffset = -cumulativePercent
                        cumulativePercent += item.percentage
                        return (
                          <circle
                            key={index}
                            cx="50"
                            cy="50"
                            r="15.915"
                            fill="none"
                            stroke={item.color}
                            strokeWidth="12"
                            strokeDasharray={strokeDasharray}
                            strokeDashoffset={strokeDashoffset}
                            className="transition-all duration-300"
                          />
                        )
                      })
                    })()}
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <div className="text-2xl font-bold text-gray-900">Total {totalStorage}</div>
                    <div className="text-sm text-gray-600">{usedStorage}</div>
                  </div>
                </div>

                {/* Legend */}
                <div className="space-y-3">
                  {storageBreakdown.map((item, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: item.color }}
                        />
                        <span className="text-sm text-gray-700">{item.category}</span>
                      </div>
                      <span className="text-sm font-medium text-gray-900">{item.percentage}%</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
