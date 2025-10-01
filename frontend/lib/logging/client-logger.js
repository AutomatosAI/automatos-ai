'use client';

// Custom error logger for client-side
export function setupErrorLogging() {
  if (typeof window !== 'undefined') {
    // Save original console methods
    const originalConsoleError = console.error;
    
    // Create a log element if it doesn't exist
    let logElement = document.getElementById('client-error-log');
    if (!logElement) {
      logElement = document.createElement('div');
      logElement.id = 'client-error-log';
      logElement.style.position = 'fixed';
      logElement.style.bottom = '0';
      logElement.style.right = '0';
      logElement.style.maxWidth = '80%';
      logElement.style.maxHeight = '50%';
      logElement.style.overflow = 'auto';
      logElement.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
      logElement.style.color = 'white';
      logElement.style.padding = '10px';
      logElement.style.zIndex = '9999';
      logElement.style.fontFamily = 'monospace';
      document.body.appendChild(logElement);
    }

    // Override console.error
    console.error = function(...args) {
      // Call original method
      originalConsoleError.apply(console, args);
      
      // Log to our custom element
      const errorMsg = args.map(arg => 
        typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
      ).join(' ');
      
      const errorDiv = document.createElement('div');
      errorDiv.style.borderBottom = '1px solid #ff6b6b';
      errorDiv.style.padding = '5px 0';
      errorDiv.textContent = `[ERROR] ${new Date().toISOString()}: ${errorMsg}`;
      logElement.appendChild(errorDiv);
    };

    // Catch unhandled errors
    window.addEventListener('error', (event) => {
      console.error('Unhandled error:', event.error);
    });

    // Catch unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      console.error('Unhandled promise rejection:', event.reason);
    });
  }
}

// Auto-invoke for immediate effect
setupErrorLogging();

export default setupErrorLogging;
