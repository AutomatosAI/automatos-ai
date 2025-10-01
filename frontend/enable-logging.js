// Add this to pages/_app.js or a utility file
const originalConsoleLog = console.log;
const originalConsoleError = console.error;
const originalConsoleWarn = console.warn;

const LOG_FILE = '/root/automatos-ai/frontend-browser.log';

// Create a logging function that writes to a file
const logToFile = (level, ...args) => {
  const timestamp = new Date().toISOString();
  const message = args.map(arg => 
    typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
  ).join(' ');
  
  const logEntry = ;
  
  // In Node.js environment only (not browser)
  if (typeof window === 'undefined') {
    const fs = require('fs');
    try {
      fs.appendFileSync(LOG_FILE, logEntry);
    } catch (err) {
      originalConsoleError('Failed to write to log file:', err);
    }
  }
};

// Override console methods
console.log = (...args) => {
  logToFile('INFO', ...args);
  originalConsoleLog.apply(console, args);
};

console.error = (...args) => {
  logToFile('ERROR', ...args);
  originalConsoleError.apply(console, args);
};

console.warn = (...args) => {
  logToFile('WARN', ...args);
  originalConsoleWarn.apply(console, args);
};

console.info = (...args) => {
  logToFile('INFO', ...args);
  originalConsoleLog.apply(console, args);
};

console.debug = (...args) => {
  logToFile('DEBUG', ...args);
  originalConsoleLog.apply(console, args);
};

// Inject into Next.js
export const enhanceLogging = () => {
  console.log('Enhanced logging enabled');
  return null;
};
