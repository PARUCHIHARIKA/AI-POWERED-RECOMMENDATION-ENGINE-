from flask import Flask, request, jsonify
from model import Recommender

app = Flask(__name__)
rec = Recommender()

@app.route('/')
def home():
    return "🎯 Recommendation Engine API is Running!"

@app.route('/recommend', methods=['GET'])
def recommend():
    movie = request.args.get('movie')
    if not movie:
        return jsonify({"error": "Movie title required!"}), 400

    results = rec.recommend(movie)
    if not results:
        return jsonify({"error": "Movie not found in dataset."}), 404

    return jsonify({
        "input": movie,
        "recommendations": results
    })

if __name__ == '__main__':
    app.run(debug=True)
