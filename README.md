# Customer Segmentation Project

## Overview
This project segments customers based on behavior and demographics using machine learning clustering techniques and Power BI visualization. The goal is to identify distinct customer groups for targeted marketing and business insights.

## Objective
- Perform clustering analysis on customer data using Python (scikit-learn)
- Analyze purchase patterns and customer preferences
- Visualize segments and key characteristics using Power BI
- Generate actionable insights for business strategy

## Key Features
- **Clustering Analysis**: Implement K-Means, Hierarchical, or other clustering algorithms
- **Data Analysis**: Analyze purchase patterns and customer preferences
- **Visualization**: Create interactive dashboards in Power BI
- **Insights**: Identify segment characteristics and target opportunities

## Expected Outcomes
- Clear customer segments with distinct behavioral patterns
- Actionable insights for targeted marketing campaigns
- Understanding of customer demographics and preferences
- Data-driven recommendations for business growth

## Tech Stack
- **Python 3.8+**
- **scikit-learn**: Machine learning clustering
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Data visualization
- **Power BI**: Interactive dashboards and reporting

## Project Structure
```
customer-segmentation/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── customer_data.csv
│   └── processed/
│       └── customer_data_processed.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_clustering_analysis.ipynb
│   └── 04_insights_and_recommendations.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── clustering.py
│   └── visualization.py
├── output/
│   ├── segment_profiles.csv
│   └── visualizations/
└── power_bi/
    └── customer_segmentation_dashboard.pbix
```

## Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/customer-segmentation.git
cd customer-segmentation
```

### Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
Place your customer data in `data/raw/customer_data.csv`

Required columns:
- Customer ID
- Age
- Annual Income
- Purchase Frequency
- Average Order Value
- Customer Tenure
- Other relevant demographics

### 2. Run Analysis
```bash
python src/data_preprocessing.py
python src/clustering.py
```

### 3. Generate Insights
Open and run the Jupyter notebooks in sequence:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 4. Visualize in Power BI
1. Open `power_bi/customer_segmentation_dashboard.pbix`
2. Connect to the processed data in `data/processed/`
3. Explore segment characteristics and trends

## Clustering Methodology

### Data Preprocessing
- Handle missing values
- Normalize numerical features
- Encode categorical variables
- Remove outliers (if applicable)

### Feature Engineering
- Create behavioral metrics
- Calculate customer lifetime value
- Generate engagement scores

### Clustering Algorithm
- **Elbow Method**: Determine optimal number of clusters
- **K-Means Clustering**: Primary segmentation technique
- **Silhouette Analysis**: Evaluate cluster quality

### Validation
- Silhouette Score
- Davies-Bouldin Index
- Business interpretability

## Key Metrics

| Metric | Purpose |
|--------|---------|
| Silhouette Score | Measures cluster cohesion and separation |
| Within-cluster sum of squares (WCSS) | Helps determine optimal cluster count |
| Davies-Bouldin Index | Lower values indicate better clustering |
| Business Metrics | Revenue per segment, customer count, churn rate |

## Results

### Sample Segments
- **High-Value Customers**: High income, frequent purchases
- **Budget-Conscious**: Lower income, price-sensitive
- **Occasional Buyers**: Infrequent purchases, lower engagement
- **Growing Customers**: Increasing purchase frequency, high potential

## Power BI Dashboard Features
- Segment distribution charts
- Customer demographic breakdowns
- Purchase behavior analysis
- Segment comparison metrics
- Trend analysis and forecasting

## File Descriptions

### Python Scripts
- `data_preprocessing.py`: Data cleaning and feature engineering
- `clustering.py`: Clustering model implementation
- `visualization.py`: Data visualization utilities

### Jupyter Notebooks
- `01_data_exploration.ipynb`: Initial data analysis and statistics
- `02_data_preprocessing.ipynb`: Data cleaning and transformation
- `03_clustering_analysis.ipynb`: Model training and evaluation
- `04_insights_and_recommendations.ipynb`: Business insights and recommendations

## Contributing
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
[Your Name/Organization]

## Contact
For questions or feedback, please reach out or open an issue on GitHub.

## Acknowledgments
- scikit-learn documentation and community
- Pandas and data science community
- Power BI best practices documentation

## Resources
- [scikit-learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Power BI Documentation](https://docs.microsoft.com/en-us/power-bi/)
- [Customer Segmentation Best Practices](https://towardsdatascience.com/customer-segmentation-5c11cfa92e50)

---

**Last Updated**: May 2026
