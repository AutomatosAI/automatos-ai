/**
 * Client-Safe Logger
 * Browser-compatible version without Node.js dependencies
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

class Logger {
  /**
   * Log a message with the specified level
   */
  private log(level: LogLevel, message: string, ...args: any[]): void {
    const timestamp = new Date().toISOString();
    console[level](`[${timestamp}] [${level.toUpperCase()}] ${message}`, ...args);
  }

  /**
   * Log a debug message
   */
  debug(message: string, ...args: any[]): void {
    this.log('debug', message, ...args);
  }

  /**
   * Log an info message
   */
  info(message: string, ...args: any[]): void {
    this.log('info', message, ...args);
  }

  /**
   * Log a warning message
   */
  warn(message: string, ...args: any[]): void {
    this.log('warn', message, ...args);
  }

  /**
   * Log an error message
   */
  error(message: string, ...args: any[]): void {
    this.log('error', message, ...args);
  }
}

// Create and export a singleton instance
export const logger = new Logger();
