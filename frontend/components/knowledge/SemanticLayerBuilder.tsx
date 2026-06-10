'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { 
  Plus, 
  TrendingUp, 
  Layers, 
  Star, 
  CheckCircle, 
  AlertCircle,
  Database,
  BarChart3,
  PieChart,
  LineChart,
  Calculator,
  Calendar,
  MapPin,
  Hash,
  Sparkles,
  Info,
  Edit,
  Trash2,
  Save
} from 'lucide-react';
import { apiClient } from '@/lib/api-client';

/**
 * Semantic Layer Builder - PROMINENT UI FEATURE
 * =============================================
 * PRD-21: Makes semantic layer management a first-class feature
 * 
 * This component allows users to visually define:
 * - Business metrics (revenue, customer count, etc.)
 * - Dimensions for grouping (time, geography, product)
 * - Relationships and hierarchies
 */

interface SemanticMetric {
  id?: string;
  name: string;
  display_name: string;
  category: string;
  sql_expression: string;
  aggregation: 'sum' | 'avg' | 'count' | 'min' | 'max';
  format: 'number' | 'currency' | 'percentage';
  description: string;
  business_definition: string;
  tables_used: string[];
  drill_down_dimensions?: string[];
  is_featured?: boolean;
  is_certified?: boolean;
  usage_count?: number;
}

interface SemanticDimension {
  id?: string;
  name: string;
  display_name: string;
  category: string;
  sql_expression: string;
  type: 'categorical' | 'temporal' | 'geographical' | 'hierarchical';
  description: string;
  hierarchy_levels?: string[];
  is_featured?: boolean;
}

interface SemanticLayerBuilderProps {
  sourceId: string;
  sourceName: string;
  dialect: string;
  onSave?: (metrics: SemanticMetric[], dimensions: SemanticDimension[]) => void;
  className?: string;
}

export function SemanticLayerBuilder({
  sourceId,
  sourceName,
  dialect,
  onSave,
  className = ''
}: SemanticLayerBuilderProps) {
  const [metrics, setMetrics] = useState<SemanticMetric[]>([]);
  const [dimensions, setDimensions] = useState<SemanticDimension[]>([]);
  const [activeTab, setActiveTab] = useState('overview');
  const [showMetricDialog, setShowMetricDialog] = useState(false);
  const [showDimensionDialog, setShowDimensionDialog] = useState(false);
  const [editingMetric, setEditingMetric] = useState<SemanticMetric | null>(null);
  const [editingDimension, setEditingDimension] = useState<SemanticDimension | null>(null);
  const [loading, setLoading] = useState(false);

  // Featured metrics that are always visible
  const featuredMetrics = metrics.filter(m => m.is_featured);
  const certifiedMetrics = metrics.filter(m => m.is_certified);

  const metricCategories = [
    { value: 'revenue', label: 'Revenue', icon: TrendingUp },
    { value: 'customers', label: 'Customers', icon: Database },
    { value: 'performance', label: 'Performance', icon: BarChart3 },
    { value: 'operations', label: 'Operations', icon: Calculator },
  ];

  const dimensionTypes = [
    { value: 'temporal', label: 'Time-based', icon: Calendar },
    { value: 'geographical', label: 'Geography', icon: MapPin },
    { value: 'categorical', label: 'Categories', icon: Layers },
    { value: 'hierarchical', label: 'Hierarchies', icon: Hash },
  ];

  useEffect(() => {
    loadSemanticLayer();
  }, [sourceId]);

  const loadSemanticLayer = async () => {
    setLoading(true);
    try {
      const data = await apiClient.get<any>(`/api/knowledge/sources/database/${sourceId}/semantic`);
      setMetrics(data.metrics || []);
      setDimensions(data.dimensions || []);
    } catch (error) {
      console.error('Failed to load semantic layer:', error);
    }
    setLoading(false);
  };

  const handleSaveMetric = (metric: SemanticMetric) => {
    if (editingMetric) {
      setMetrics(metrics.map(m => m.id === editingMetric.id ? metric : m));
    } else {
      setMetrics([...metrics, { ...metric, id: Date.now().toString() }]);
    }
    setShowMetricDialog(false);
    setEditingMetric(null);
  };

  const handleSaveDimension = (dimension: SemanticDimension) => {
    if (editingDimension) {
      setDimensions(dimensions.map(d => d.id === editingDimension.id ? dimension : d));
    } else {
      setDimensions([...dimensions, { ...dimension, id: Date.now().toString() }]);
    }
    setShowDimensionDialog(false);
    setEditingDimension(null);
  };

  const handleSaveAll = async () => {
    if (onSave) {
      onSave(metrics, dimensions);
    } else {
      // Default save to API
      try {
        await apiClient.post(`/api/knowledge/sources/database/${sourceId}/semantic`, { metrics, dimensions });
      } catch (error) {
        console.error('Failed to save semantic layer:', error);
      }
    }
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header with prominent branding */}
      <Card className="border-2 border-primary/20 bg-gradient-to-r from-primary/5 to-transparent">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Sparkles className="w-8 h-8 text-primary" />
              <div>
                <CardTitle className="text-2xl">Semantic Layer</CardTitle>
                <CardDescription className="text-base mt-1">
                  Define business logic once, use everywhere • {sourceName}
                </CardDescription>
              </div>
            </div>
            <Button onClick={handleSaveAll} size="lg">
              <Save className="w-4 h-4 mr-2" />
              Save Changes
            </Button>
          </div>
        </CardHeader>
      </Card>

      {/* Overview stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Metrics</p>
                <p className="text-2xl font-bold">{metrics.length}</p>
              </div>
              <Calculator className="w-8 h-8 text-muted-foreground/50" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Dimensions</p>
                <p className="text-2xl font-bold">{dimensions.length}</p>
              </div>
              <Layers className="w-8 h-8 text-muted-foreground/50" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Featured</p>
                <p className="text-2xl font-bold">{featuredMetrics.length}</p>
              </div>
              <Star className="w-8 h-8 text-warning/50" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Certified</p>
                <p className="text-2xl font-bold">{certifiedMetrics.length}</p>
              </div>
              <CheckCircle className="w-8 h-8 text-success/50" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main content tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview" className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="metrics" className="flex items-center gap-2">
            <Calculator className="w-4 h-4" />
            Metrics ({metrics.length})
          </TabsTrigger>
          <TabsTrigger value="dimensions" className="flex items-center gap-2">
            <Layers className="w-4 h-4" />
            Dimensions ({dimensions.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Featured Metrics Section */}
          {featuredMetrics.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Star className="w-5 h-5 text-warning" />
                  Featured Metrics
                </CardTitle>
                <CardDescription>
                  Most commonly used metrics for this database
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {featuredMetrics.map((metric) => (
                    <Card key={metric.id} className="border-2 border-warning/20">
                      <CardContent className="pt-4">
                        <div className="flex items-start justify-between">
                          <div>
                            <h4 className="font-semibold">{metric.display_name}</h4>
                            <p className="text-sm text-muted-foreground mt-1">
                              {metric.business_definition}
                            </p>
                            <div className="flex items-center gap-2 mt-2">
                              <Badge variant="secondary">{metric.aggregation}</Badge>
                              <Badge variant="outline">{metric.format}</Badge>
                              {metric.is_certified && (
                                <CheckCircle className="w-4 h-4 text-success" />
                              )}
                            </div>
                          </div>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => {
                              setEditingMetric(metric);
                              setShowMetricDialog(true);
                            }}
                          >
                            <Edit className="w-4 h-4" />
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Quick Start Guide */}
          <Card>
            <CardHeader>
              <CardTitle>Quick Start Guide</CardTitle>
              <CardDescription>
                Set up your semantic layer in minutes
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Alert>
                  <Info className="w-4 h-4" />
                  <AlertDescription>
                    The semantic layer lets you define business metrics and dimensions once,
                    then use them everywhere - in natural language queries, dashboards, and reports.
                  </AlertDescription>
                </Alert>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card className="cursor-pointer hover:border-primary" 
                        onClick={() => {
                          setActiveTab('metrics');
                          setShowMetricDialog(true);
                        }}>
                    <CardContent className="pt-4">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-primary/10 rounded-lg">
                          <Calculator className="w-6 h-6 text-primary" />
                        </div>
                        <div>
                          <h4 className="font-semibold">Add a Metric</h4>
                          <p className="text-sm text-muted-foreground">
                            Define calculations like revenue, user count
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card className="cursor-pointer hover:border-primary"
                        onClick={() => {
                          setActiveTab('dimensions');
                          setShowDimensionDialog(true);
                        }}>
                    <CardContent className="pt-4">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-primary/10 rounded-lg">
                          <Layers className="w-6 h-6 text-primary" />
                        </div>
                        <div>
                          <h4 className="font-semibold">Add a Dimension</h4>
                          <p className="text-sm text-muted-foreground">
                            Create groupings like time, region, category
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="metrics" className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold">Business Metrics</h3>
            <Button onClick={() => setShowMetricDialog(true)}>
              <Plus className="w-4 h-4 mr-2" />
              Add Metric
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {metrics.map((metric) => (
              <Card key={metric.id}>
                <CardContent className="pt-4">
                  <div className="space-y-2">
                    <div className="flex items-start justify-between">
                      <h4 className="font-semibold">{metric.display_name}</h4>
                      <div className="flex items-center gap-1">
                        {metric.is_featured && <Star className="w-4 h-4 text-warning" />}
                        {metric.is_certified && <CheckCircle className="w-4 h-4 text-success" />}
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => {
                            setEditingMetric(metric);
                            setShowMetricDialog(true);
                          }}
                        >
                          <Edit className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground">{metric.description}</p>
                    <div className="flex items-center gap-2">
                      <Badge>{metric.category}</Badge>
                      <Badge variant="outline">{metric.aggregation}</Badge>
                      <Badge variant="secondary">{metric.format}</Badge>
                    </div>
                    <div className="pt-2 border-t">
                      <code className="text-xs bg-muted px-2 py-1 rounded">
                        {metric.sql_expression}
                      </code>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="dimensions" className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold">Dimensions</h3>
            <Button onClick={() => setShowDimensionDialog(true)}>
              <Plus className="w-4 h-4 mr-2" />
              Add Dimension
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {dimensions.map((dimension) => (
              <Card key={dimension.id}>
                <CardContent className="pt-4">
                  <div className="space-y-2">
                    <div className="flex items-start justify-between">
                      <h4 className="font-semibold">{dimension.display_name}</h4>
                      <div className="flex items-center gap-1">
                        {dimension.is_featured && <Star className="w-4 h-4 text-warning" />}
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => {
                            setEditingDimension(dimension);
                            setShowDimensionDialog(true);
                          }}
                        >
                          <Edit className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground">{dimension.description}</p>
                    <div className="flex items-center gap-2">
                      <Badge>{dimension.type}</Badge>
                      <Badge variant="outline">{dimension.category}</Badge>
                    </div>
                    {dimension.hierarchy_levels && (
                      <div className="pt-2 text-sm">
                        Hierarchy: {dimension.hierarchy_levels.join(' → ')}
                      </div>
                    )}
                    <div className="pt-2 border-t">
                      <code className="text-xs bg-muted px-2 py-1 rounded">
                        {dimension.sql_expression}
                      </code>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>

      {/* Metric Dialog */}
      <MetricDialog
        open={showMetricDialog}
        onClose={() => {
          setShowMetricDialog(false);
          setEditingMetric(null);
        }}
        onSave={handleSaveMetric}
        metric={editingMetric}
        dialect={dialect}
      />

      {/* Dimension Dialog */}
      <DimensionDialog
        open={showDimensionDialog}
        onClose={() => {
          setShowDimensionDialog(false);
          setEditingDimension(null);
        }}
        onSave={handleSaveDimension}
        dimension={editingDimension}
        dialect={dialect}
      />
    </div>
  );
}

// Metric creation/editing dialog
function MetricDialog({ 
  open, 
  onClose, 
  onSave, 
  metric, 
  dialect 
}: {
  open: boolean;
  onClose: () => void;
  onSave: (metric: SemanticMetric) => void;
  metric: SemanticMetric | null;
  dialect: string;
}) {
  const [formData, setFormData] = useState<SemanticMetric>(
    metric || {
      name: '',
      display_name: '',
      category: 'revenue',
      sql_expression: '',
      aggregation: 'sum',
      format: 'number',
      description: '',
      business_definition: '',
      tables_used: [],
      is_featured: false,
      is_certified: false
    }
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(formData);
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            {metric ? 'Edit Metric' : 'Create New Metric'}
          </DialogTitle>
          <DialogDescription>
            Define a business metric that can be used across all queries and reports
          </DialogDescription>
        </DialogHeader>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="name">Internal Name</Label>
              <Input
                id="name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                placeholder="total_revenue"
                required
              />
            </div>
            
            <div>
              <Label htmlFor="display_name">Display Name</Label>
              <Input
                id="display_name"
                value={formData.display_name}
                onChange={(e) => setFormData({ ...formData, display_name: e.target.value })}
                placeholder="Total Revenue"
                required
              />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div>
              <Label htmlFor="category">Category</Label>
              <Select
                value={formData.category}
                onValueChange={(v) => setFormData({ ...formData, category: v })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="revenue">Revenue</SelectItem>
                  <SelectItem value="customers">Customers</SelectItem>
                  <SelectItem value="performance">Performance</SelectItem>
                  <SelectItem value="operations">Operations</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="aggregation">Aggregation</Label>
              <Select
                value={formData.aggregation}
                onValueChange={(v: any) => setFormData({ ...formData, aggregation: v })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="sum">Sum</SelectItem>
                  <SelectItem value="avg">Average</SelectItem>
                  <SelectItem value="count">Count</SelectItem>
                  <SelectItem value="min">Minimum</SelectItem>
                  <SelectItem value="max">Maximum</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="format">Format</Label>
              <Select
                value={formData.format}
                onValueChange={(v: any) => setFormData({ ...formData, format: v })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="number">Number</SelectItem>
                  <SelectItem value="currency">Currency</SelectItem>
                  <SelectItem value="percentage">Percentage</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div>
            <Label htmlFor="sql_expression">SQL Expression</Label>
            <Textarea
              id="sql_expression"
              value={formData.sql_expression}
              onChange={(e) => setFormData({ ...formData, sql_expression: e.target.value })}
              placeholder="SUM(orders.amount)"
              className="font-mono text-sm"
              rows={3}
              required
            />
            <p className="text-xs text-muted-foreground mt-1">
              SQL expression for {dialect} database
            </p>
          </div>

          <div>
            <Label htmlFor="business_definition">Business Definition</Label>
            <Textarea
              id="business_definition"
              value={formData.business_definition}
              onChange={(e) => setFormData({ ...formData, business_definition: e.target.value })}
              placeholder="Total revenue from all completed orders"
              rows={2}
            />
          </div>

          <div className="flex items-center space-x-4">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={formData.is_featured}
                onChange={(e) => setFormData({ ...formData, is_featured: e.target.checked })}
              />
              <span className="text-sm flex items-center gap-1">
                <Star className="w-4 h-4 text-warning" />
                Featured
              </span>
            </label>
            
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={formData.is_certified}
                onChange={(e) => setFormData({ ...formData, is_certified: e.target.checked })}
              />
              <span className="text-sm flex items-center gap-1">
                <CheckCircle className="w-4 h-4 text-success" />
                Certified
              </span>
            </label>
          </div>

          <DialogFooter>
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit">
              {metric ? 'Update' : 'Create'} Metric
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

// Dimension creation/editing dialog
function DimensionDialog({ 
  open, 
  onClose, 
  onSave, 
  dimension,
  dialect 
}: {
  open: boolean;
  onClose: () => void;
  onSave: (dimension: SemanticDimension) => void;
  dimension: SemanticDimension | null;
  dialect: string;
}) {
  const [formData, setFormData] = useState<SemanticDimension>(
    dimension || {
      name: '',
      display_name: '',
      category: '',
      sql_expression: '',
      type: 'categorical',
      description: '',
      hierarchy_levels: [],
      is_featured: false
    }
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(formData);
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            {dimension ? 'Edit Dimension' : 'Create New Dimension'}
          </DialogTitle>
          <DialogDescription>
            Define a dimension for grouping and filtering data
          </DialogDescription>
        </DialogHeader>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="name">Internal Name</Label>
              <Input
                id="name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                placeholder="customer_region"
                required
              />
            </div>
            
            <div>
              <Label htmlFor="display_name">Display Name</Label>
              <Input
                id="display_name"
                value={formData.display_name}
                onChange={(e) => setFormData({ ...formData, display_name: e.target.value })}
                placeholder="Customer Region"
                required
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="type">Type</Label>
              <Select
                value={formData.type}
                onValueChange={(v: any) => setFormData({ ...formData, type: v })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="temporal">Temporal (Time-based)</SelectItem>
                  <SelectItem value="geographical">Geographical</SelectItem>
                  <SelectItem value="categorical">Categorical</SelectItem>
                  <SelectItem value="hierarchical">Hierarchical</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="category">Category</Label>
              <Input
                id="category"
                value={formData.category}
                onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                placeholder="customer, product, time"
              />
            </div>
          </div>

          <div>
            <Label htmlFor="sql_expression">SQL Expression</Label>
            <Textarea
              id="sql_expression"
              value={formData.sql_expression}
              onChange={(e) => setFormData({ ...formData, sql_expression: e.target.value })}
              placeholder="customers.region"
              className="font-mono text-sm"
              rows={2}
              required
            />
            <p className="text-xs text-muted-foreground mt-1">
              SQL expression for {dialect} database
            </p>
          </div>

          <div>
            <Label htmlFor="description">Description</Label>
            <Input
              id="description"
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              placeholder="Geographic region of the customer"
            />
          </div>

          <div className="flex items-center">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={formData.is_featured}
                onChange={(e) => setFormData({ ...formData, is_featured: e.target.checked })}
              />
              <span className="text-sm flex items-center gap-1">
                <Star className="w-4 h-4 text-warning" />
                Featured
              </span>
            </label>
          </div>

          <DialogFooter>
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit">
              {dimension ? 'Update' : 'Create'} Dimension
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
