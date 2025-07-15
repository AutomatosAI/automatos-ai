
from flask import Flask, render_template, jsonify
import requests
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Sample cryptocurrency data (in production, this would come from an API)
CRYPTO_DATA = {
    'bitcoin': {'price': 45000, 'change': 2.5},
    'ethereum': {'price': 3200, 'change': -1.2},
    'cardano': {'price': 0.85, 'change': 5.8}
}

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('dashboard.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'crypto-web-server'
    })

@app.route('/api/prices')
def get_prices():
    """Get cryptocurrency prices"""
    try:
        logger.info("Fetching cryptocurrency prices")
        return jsonify({
            'success': True,
            'data': CRYPTO_DATA,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error fetching prices: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/price/<crypto>')
def get_price(crypto):
    """Get price for specific cryptocurrency"""
    try:
        crypto = crypto.lower()
        if crypto in CRYPTO_DATA:
            return jsonify({
                'success': True,
                'crypto': crypto,
                'data': CRYPTO_DATA[crypto],
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Cryptocurrency not found'
            }), 404
    except Exception as e:
        logger.error(f"Error fetching price for {crypto}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
