# ✈️ SkyPrice AI: Decoding Flight Prices with Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20LightGBM-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%7C%20Keras-red)
![Optimization](https://img.shields.io/badge/Optimization-Optuna-green)

Have you ever wondered why a flight costs $200 on a Tuesday but jumps to $600 by Thursday? I did too. 

Flight pricing is notoriously dynamic, heavily skewed, and often feels completely random to travelers. I built **SkyPrice AI** to figure out exactly what drives that volatility. Instead of relying on static rules or simple averages, I used advanced machine learning to map out the biological booking rhythms, physical route constraints, and temporal market surges that airlines use to price their tickets. 

---

## 🗄️ The Data: No Pre-Packaged Datasets Here
I didn't want to use a clean, outdated Kaggle dataset. To build a model that understands the *real* market, I had to source the data myself. 

* **The Pipeline:** I built a custom data extraction pipeline using a Python scraper and local Playwright compute to bypass proxies . 
* **The Scope:** I pulled over **100,000 live flight records** targeting 8 major US domestic hubs (JFK, LAX, ORD, etc.), forecasting up to 120 days into the future .
* **The Result:** A highly optimized ~15.0 MB dataset containing 24 core physical and temporal features.

---

## ⚙️ Under the Hood: Feature Engineering
Raw data isn't enough to capture complex market swings.I expanded the baseline dataset by engineering **18 custom mathematical signals**

* **Cyclical Time:** I used sine and cosine transformations to map time . This helps the AI understand that Sunday night wraps around seamlessly into Monday morning, rather than treating them as a linear line. 
* **Interaction Terms:** I created compounded features (like `Dist_x_Stops` and `Weekend_Holiday`) to isolate specific surge effects.
* **Leak-Proof Encoding:** To prevent the model from just "memorizing" expensive routes or airlines, I completely avoided naive target encoding. Instead, I implemented **K-Fold Target Encoding**. This ensures the algorithm actually learns the underlying patterns without data leakage.

---

## 🧠 The Algorithm Arena 

I started by putting four baseline heavyweights into the arena: Random Forest, CatBoost, LightGBM, and XGBoost . XGBoost and LightGBM emerged as the top contenders.

### Bayesian Optimization
To get the absolute best performance, I didn't just blindly guess the hyperparameter settings. I deployed **Optuna** to run 50 Bayesian optimization trials.By using Tree-structured Parzen Estimators (TPE), the algorithm intelligently navigated learning rates, maximum tree depths, and L1/L2 regularization to find the mathematically optimal configurations .

### 🏆 The Ultimate Blend
The overall winner wasn't a single model.By blending my tuned XGBoost (54%) and LightGBM (46%) models together, I created an optimized ensemble that captured market realities neither model could find alone.

* **$R^2$ Score:** **0.8019** (Predicting over 80% of the market variance!
* **Mean Absolute Error (MAE):** **$45.90** 
* **RMSE:** **$70.53** 

### 🔬 Pushing the Limits: Deep Learning
Just to see how far I could push it, I also built a custom 4-Layer Deep Manifold Neural Network (MLP Regressor) as a non-linear counterpart to the tree ensembles. I used **Swish activations** to handle complex non-linear mapping and a **Huber Loss Function** because it is mathematically robust against those rare, crazy $9,000 ticket anomalies.It achieved a highly competitive $49.12 MAE!

---

## 📊 What Did the AI Actually Learn?
An 80% accuracy score is great, but the real value is in *how* the model makes its decisions. By analyzing the feature importance, the data proved:

* **Convenience > Distance:** You'd think physical mileage dictates the price, but it doesn't.The AI proved that convenience factors (specifically `Is_Nonstop` and `Arrives_Next_Day`) heavily outrank raw distance in driving price variations .
* **The "Golden Window":** Prices don't just go up linearly. They actually stabilize in a sweet spot 21 to 45 days before departure. 
* **The Urgency Penalty:** If you book less than 14 days out, you trigger extreme inelastic surge pricing. 
* **The Sunday Surge:** Sunday is consistently the most expensive day to fly, while Tuesday and Wednesday offer the deepest market discounts.

---
## 🤝 Let's Connect & Collaborate!

Thank you so much for checking out SkyPrice AI! Building this was a massive learning experience—from wrangling the Playwright data scrapers to watching model finally converge. 

If you found this project interesting, have feedback on the methodology, or just want to chat about data science and machine learning, I would absolutely love to hear from you. 

* **Drop a ⭐** on this repository if you found the code or methodology helpful!
* **Issues & Pull Requests** are always welcome if you want to experiment with the dataset or tweak the ensemble blend.

Thanks for stopping by, and happy coding.

