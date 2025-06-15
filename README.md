# ANIME RECOMMENDER

This recommender is developed using user-based collaborative filtering and SVM classifier. The data is acquired from [kaggle](https://www.kaggle.com/datasets/azathoth42/myanimelist).

The model is trained by using `1000 user` at max, because hardware limitation.  Even if it's only `1000 user`, the preprocessed input's dimension are  somewhere around `390785x3534`, which is very huge considering the fact that the exported CSV size is reaching `±5 GB` in total.

The way this recommendation system works is by using cosine similarity to find `k` similar user and decide the top `n` anime based on each similar user scoring. The SVM classifier predict the selected user's disliked animes based on each anime's genre and then remove it from the recommendation list.

## 🚀 Features
- User-based collaborative filtering with SVM classifier
- Web interface for easy interaction
- CLI support for direct recommendations
- Genre-based filtering and analysis
- Cosine similarity for user matching
- Handles large-scale anime dataset

## 🛠️ Setup & Installation

1. Clone this repo & move to its directory
```bash
git clone <repository-url>
cd anime-recommender
```

2. Create and activate your virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac
```

3. Install the required packages
```bash
pip install -r requirements.txt
```

## 🎮 How To Use

### 🌐 Webserver Method
1. Configure environment
```bash
cp .env.example .env
# Edit .env with your settings
```

2. Start the Flask server
```bash
flask run
```

3. Open your browser and navigate to `http://localhost:5000`
4. Select a user and wait for recommendations

### 💻 CLI Method
1. Follow setup instructions in [DATA](./data/README.md) directory
2. Follow setup instructions in [EXPORT](./export/README.md) directory
3. Run the recommender
```bash
python recommender.py
```

> **Note**: By default, it shows recommendations for user [`Zexu`](https://myanimelist.net/profile/Zexu) (user_id: `459521`). Edit the user in recommender.py to get recommendations for different users.

## 🔧 Project Structure
```
anime-recommender/
├── app/
│   ├── controller/
│   ├── model/
│   ├── static/
│   └── templates/
├── data/
├── export/
└── recommender.py
```

## 🤝 Contributing
Feel free to open issues and pull requests!

## 📄 License
This project is open source and available under the MIT License.
