/**
 * Feature flags configuration for Automatos AI Platform
 * 
 * Use this file to enable/disable features during development or in production
 */

const FeatureFlags = {
  // Dashboard features
  enableDashboardRealtime: true,
  enableDashboardMetrics: true,
  
  // Chatbot features
  enableChatbot: false, // Set to false to disable the chatbot widget
  enableChatbotSuggestions: false,
  
  // Agent features
  enableAgentCollaboration: true,
  enableAgentLearning: true,
  
  // Workflow features
  enableWorkflowEditor: true,
  enableWorkflowPreview: true,
  
  // Document features
  enableDocumentSearch: true,
  enableDocumentPreview: true,
  
  // Analytics features
  enableAnalytics: true,
  enablePerformanceCharts: true,
  
  // System features
  enableDebugMode: false,
  enableSystemMonitoring: true
};

module.exports = FeatureFlags;


