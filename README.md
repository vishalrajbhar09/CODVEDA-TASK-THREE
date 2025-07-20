Task 3: Clustering (Unsupervised Learning)

Intern Name: Vishal Ramkumar Rajbhar

Intern ID: CV/A1/18203

Domain: Data Science

Program: Code Veda Intermediate Internship

📝 Task Overview
This project demonstrates the application of unsupervised learning techniques to discover patterns and structure within unlabeled data. A synthetic dataset simulating cricket player statistics is used to group players into clusters based on their performance metrics using the K-Means clustering algorithm.

🎯 Objectives
Create a custom cricket performance dataset with realistic player statistics.

Apply K-Means clustering to segment players into performance-based groups.

Determine the optimal number of clusters using the Elbow Method and Silhouette Score.

Visualize clustering results using Principal Component Analysis (PCA) for dimensionality reduction.

Interpret and summarize cluster insights.

📁 Dataset Description
Type: Synthetic dataset (generated programmatically)

Size: 200 players


<img width="606" height="381" alt="image" src="https://github.com/user-attachments/assets/d4a6d190-687c-4193-99f1-bd4274a2c811" />

<img width="583" height="382" alt="image" src="https://github.com/user-attachments/assets/91ffc981-5b0a-47b8-a648-27bd29fc29c0" />



Features:

Batting_Avg: Player’s batting average

Strike_Rate: Batting strike rate

Wickets: Number of wickets taken

Economy: Bowling economy rate

These features represent a mix of batting and bowling metrics to enable meaningful clustering of batsmen, bowlers, all-rounders, and lower performers.

🧪 Techniques & Algorithms
K-Means Clustering

Principal Component Analysis (PCA) for 2D visualization

Cluster Evaluation:

Inertia (WCSS) via Elbow Method

Silhouette Score for cluster cohesion and separation

📈 Evaluation Metrics
Elbow Method: Assesses the optimal k by identifying the point where adding more clusters yields diminishing returns.

Silhouette Score: Measures how well each point fits within its cluster vs. others, with a higher score indicating better-defined clusters.

📊 Visualizations
Elbow Curve for optimal cluster count

Silhouette Score Plot across different k values

2D Cluster Visualization using PCA (Principal Component Analysis)

FINAL OUTPUT*
<img width="580" height="487" alt="image" src="https://github.com/user-attachments/assets/fc58fe5f-d816-4690-8430-e2fbf1977922" />


🛠️ Tools & Technologies
Language: Python

Libraries:

pandas, numpy – Data creation and manipulation

scikit-learn – Clustering, PCA, evaluation

matplotlib, seaborn – Data visualization

📌 Key Takeaways
Learned how to simulate realistic domain-specific datasets (e.g., cricket).

Applied K-Means to uncover patterns in unlabeled data.

Practiced model validation techniques (elbow method and silhouette score).

Developed skill in interpreting and visualizing clusters through PCA.
