# Leafy_Hospital_Microsoft_genAI_resources
Repo to share the resources used in Leafy Hospital - Microsoft genAI showcase

The repo consists of the artifacts that were mentioned in the solution "AI-Powered Healthcare with MongoDB & Microsoft".

MongoDB Atlas and Microsoft AI technologies converge in an innovative healthcare solution called "Leafy Hospital," showcasing how cutting-edge technology can transform breast cancer diagnosis and patient care. This integrated system leverages MongoDB's flexible data platform for unifying operational, metadata, and AI data, while incorporating Microsoft's advanced capabilities including Azure OpenAI, Microsoft Fabric, and Power BI to create a comprehensive healthcare analytics and diagnostic solution.

The solution demonstrates three key technological approaches:

**Predictive AI for early detection** using deep learning models to analyze mammograms and predict BI-RADS scores;
  * The repo includes two models that can be trained and used to predict BI-RADS scores (*BIRADS_Prediction_ML_new.ipynb*) and to classify the cancer as malignant or benign (*classification_model_M_B_new.ipynb*).
  * The dataset used for the BI-RADS scoring was taken from Kaggle link [here](https://www.kaggle.com/datasets/asmaasaad/king-abdulaziz-university-mammogram-dataset). The dataset contains 1416 cases; all cases include images with two types of views (CC and MLO) for both breasts (right and left). The dataset was classified into 1 to 5 categories in accordance with BI-RADS .
  * The dataset for the Malignant(M) or Benign(B) classification model was taken from Kaggle link [here](https://www.kaggle.com/datasets/ninjacoding/breast-cancer-wisconsin-benign-or-malignant). It takes 9 parameters and predicts the Class as 2 for Benign and 4 for Malignant.

**Generative AI for workflow automation,** featuring vector search capabilities and RAG-based chatbots for intelligent information retrieval;
  * The repo includes the chatbot code (*chatbot_new.py*) which takes three different contexts into consideration
      * The chatbot code has been modified to work as a standalone python code. Thus, hardcodings are used. It uses three different collections to give three different contexts to the chatbot.
      * RAG_collection consists of cancer related guides and documents which is stored as vectorised chunks using the *pdfDataBot.py* code. The chunks similar to the question asked are vector searched and provided as "Context section" to the chatbot.
      * Patients collection contains patients basic information like name, age etc. which is directly queried using MongoDB Query Language (MQL) and presented to chatbot as "Patient Info".
      * Reports collection contains the details of each visit of the patient and the details gathered during the same. The last 10 visits information is summarised using LLM and provided to the chatbot as "Patient history".
  * The repo also includes the code to add documentation to MongoDB Atlas by chunking, vectorising the chunks and inserting into a MongoDB collection
      * The *pdfDataBot.py* is a streamlit application and needs to be invoked by running the command - *"streamlit run pdfDataBot.py"*

**Advanced Analytics** that combines real-time operational insights with long-term trend analysis through Power BI integration. ThePower BI file *"LeafyHospitalBI.pbix"* is added under Releases -> 1 tags on the right side menu of the repo. Click on v1.0 under the tags to find the pbix file.




