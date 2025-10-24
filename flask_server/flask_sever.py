# flask_server.py
"""
Flask REST API Server - provides data endpoints
Install dependencies: pip install flask
"""

from flask import Flask, jsonify, request
from datetime import datetime
import random

app = Flask(__name__)

# Sample data
users_db = [
    {"id": 1, "name": "Alice Johnson", "email": "alice@example.com", "age": 28},
    {"id": 2, "name": "Bob Smith", "email": "bob@example.com", "age": 34},
    {"id": 3, "name": "Charlie Brown", "email": "charlie@example.com", "age": 25},
]

products_db = [
    {"id": 1, "name": "Laptop", "price": 999.99, "stock": 15},
    {"id": 2, "name": "Mouse", "price": 29.99, "stock": 50},
    {"id": 3, "name": "Keyboard", "price": 79.99, "stock": 30},
    {"id": 4, "name": "Monitor", "price": 299.99, "stock": 20},
]

@app.route('/')
def home():
    """API documentation"""
    return jsonify({
        "message": "Flask REST API Server",
        "endpoints": {
            "/api/users": "GET - List all users",
            "/api/users/<id>": "GET - Get user by ID",
            "/api/products": "GET - List all products",
            "/api/products/<id>": "GET - Get product by ID",
            "/api/weather": "GET - Get current weather (mock data)",
            "/api/time": "GET - Get current server time",
            "/api/random": "GET - Get random number (params: min, max)"
        }
    })

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users"""
    return jsonify({"users": users_db, "count": len(users_db)})

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get user by ID"""
    user = next((u for u in users_db if u["id"] == user_id), None)
    if user:
        return jsonify(user)
    return jsonify({"error": "User not found"}), 404

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get all products"""
    return jsonify({"products": products_db, "count": len(products_db)})

@app.route('/api/products/<int:product_id>', methods=['GET'])
def get_product(product_id):
    """Get product by ID"""
    product = next((p for p in products_db if p["id"] == product_id), None)
    if product:
        return jsonify(product)
    return jsonify({"error": "Product not found"}), 404

@app.route('/api/weather', methods=['GET'])
def get_weather():
    """Get current weather (mock data)"""
    cities = ["New York", "London", "Tokyo", "Paris", "Sydney"]
    conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy", "Windy"]
    
    city = request.args.get('city', random.choice(cities))
    
    return jsonify({
        "city": city,
        "temperature": random.randint(10, 30),
        "condition": random.choice(conditions),
        "humidity": random.randint(30, 90),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/time', methods=['GET'])
def get_time():
    """Get current server time"""
    return jsonify({
        "time": datetime.now().strftime("%H:%M:%S"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/random', methods=['GET'])
def get_random():
    """Get random number"""
    min_val = int(request.args.get('min', 1))
    max_val = int(request.args.get('max', 100))
    
    return jsonify({
        "number": random.randint(min_val, max_val),
        "min": min_val,
        "max": max_val
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Flask REST API Server")
    print("=" * 60)
    print("\nStarting server on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  - GET  /api/users")
    print("  - GET  /api/users/<id>")
    print("  - GET  /api/products")
    print("  - GET  /api/products/<id>")
    print("  - GET  /api/weather?city=<city>")
    print("  - GET  /api/time")
    print("  - GET  /api/random?min=<min>&max=<max>")
    print("\nPress CTRL+C to stop\n")
    print("=" * 60)
    
    app.run(debug=True, port=5000)