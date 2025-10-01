/**
 * Browser-compatible logger
 */

class Logger {
  info(message, ...args) {
    console.info(`[INFO] ${message}`, ...args);
  }
  
  debug(message, ...args) {
    console.debug(`[DEBUG] ${message}`, ...args);
  }
  
  warn(message, ...args) {
    console.warn(`[WARN] ${message}`, ...args);
  }
  
  error(message, ...args) {
    console.error(`[ERROR] ${message}`, ...args);
  }
}

const logger = new Logger();
module.exports = logger;
module.exports.default = Logger;
