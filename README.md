# Apply Machine Learning in the Company to Predict the Quality of Sales Leads

This repository is part of the thesis of the Master's in Data Science by the University of Barcelona 2019/2020.

This work has been done with the collaboration of the EMEA 3D sales team in HP. The dataset in which the project is based on is currently more than 40,500 leads from the fiscal year 2016 quarter 1 to the fiscal year 2020 quarter 4. On average, every week 257 new leads are entered into the system.

The main objective of this project is to develop a data science pipeline, capable of predicting, for every lead, the quality of it. Meaning quality, the probability in which the lead is going to become a possible sell, and to advance to the next sale stages. By doing this, we want to **achieve a transition from decisions based on intuition from the salesman, to more data-driven decision-making with the use of a score, from 0 to 1, that will be an indication of the quality of the lead.**

The pipeline developed is widely explained in the thesis file found in this repository. But the main structure of the pipeline is exposed below:

![Image of pipeline](https://github.com/jordisc97/Apply-Machine-Learning-in-the-Company-to-Predict-the-Quality-of-Sales-Leads/blob/master/Pipeline_Project.png)

The pipeline consist of the following processes:

* **Joining:** First joining all the sources in one table, this is done in a Z8 server inside HP.
* **Scraping:**  Web scraping techniques were used to enhance the information received from the company CRM.
* **Preprocessing:** Some of the scraped pages were not in English and hence, they needed to be translated. Along with the translation, other tasks such as data cleaning, feature engineering, and encoding were needed to assure the best possible dataset to feed the algorithm.
* **Training:** The model can be periodically retrained with a pipeline parameter. This training was developed on all the data to predict the score for just the leads that were not assigned to any state yet. The algorithm selected to do the predictions was an Extreme Gradient Boosting. The output of this training is a pickle file used to do faster predictions.
* **Prediction:** In this step, the score for each lead was output along with the explainability of the most important attribute for the decision with the LIME package.
* **PowerBi:** The file resulting from the prediction was retrieved from the server and put to a PowerBi so the whole organization can use the data from the scarping, scoring, and explainability algorithms.
