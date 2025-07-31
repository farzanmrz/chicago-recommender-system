from lenskit.pipeline import topn_pipeline
from lenskit.basic import PopScorer, BiasScorer
from lenskit.als import ImplicitMFScorer, BiasedMFScorer
from lenskit.metrics import RunAnalysis, RMSE, RBP, NDCG, RecipRank
from lenskit import batch
from lenskit.knn.user import UserKNNScorer
from lenskit.knn.item import ItemKNNScorer

N = 50

def biasModelTrainer(train_data, test_data):
    # Create an instance of the model object
    bias_model = BiasScorer()

    # Wrap the instance in a pipeline (topn_pipeline(model))
    bias_pipe = topn_pipeline(bias_model, predicts_ratings=True)

    # Train the algorithm on the training data
    bias_pipe.train(train_data)

    # For a prediction algorithm (P above): generate predictions for all test useritem pairs using batch.predict.
    bias_preds = batch.predict(bias_pipe, test_data)

    # Generate 50-item recommendation lists for test users with batch.recommend.
    bias_recs = batch.recommend(bias_pipe, test_data, n=N)
    bias_recs.to_csv("bias_recommendations.csv", index=False)


    # Measure the recommendation quality (using RunAnalysis).
    ra1 = RunAnalysis()
    ra1.add_metric(NDCG(k=N))
    bias_results1 = ra1.compute(bias_recs, test_data)
    # print(bias_results1.list_summary())
    
    ra2 = RunAnalysis()
    ra2.add_metric(RMSE())
    bias_pred_results = ra2.compute(bias_preds, test_data)
    # print(bias_pred_results.list_summary())
    return bias_results1.list_summary(), bias_pred_results.list_summary()

def popularModelTrainer(split):
    pop_model = PopScorer()
    pop_pipe = topn_pipeline(pop_model)
    pop_pipe.train(split.train)

    pop_recs = batch.recommend(pop_pipe, split.test, n=N)
    pop_recs.to_csv("pop_recommendations.csv", index=False)


    ra = RunAnalysis()
    ra.add_metric(NDCG(k=N))
    pop_results = ra.compute(pop_recs, split.test)
    # print(pop_results.list_summary())
    return pop_results.list_summary()

def ExplicitMFModelTrainer(split):
    als_model = BiasedMFScorer(features=50)
    als_pipe = topn_pipeline(als_model, predicts_ratings=True)
    als_pipe.train(split.train)

    als_preds = batch.predict(als_pipe, split.test)
    als_recs = batch.recommend(als_pipe, split.test, n=N)
    als_recs.to_csv("als_recommendations.csv", index=False)


    # Measure the recommendation quality (using RunAnalysis).
    ra1 = RunAnalysis()
    ra1.add_metric(NDCG(k=N))
    als_results1 = ra1.compute(als_recs, split.test)
    # print(als_results1.list_summary())
    
    ra2 = RunAnalysis()
    ra2.add_metric(RMSE())
    als_pred_results = ra2.compute(als_preds, split.test)
    # print(als_pred_results.list_summary())
    return als_results1.list_summary(), als_pred_results.list_summary()
    

def ImplicitMFModelTrainer(split):
    imf_model = ImplicitMFScorer(features=50)
    imf_pipe = topn_pipeline(imf_model)
    imf_pipe.train(split.train)

    imf_recs = batch.recommend(imf_pipe, split.test, n=N)
    imf_recs.to_csv("imf_recommendations.csv", index=False)


    ra = RunAnalysis()
    ra.add_metric(NDCG(k=N))
    imf_results = ra.compute(imf_recs, split.test)
    # print(imf_results.list_summary())
    return imf_results.list_summary()

def UserUserKnnModelTrainer(split):
    knnUU_model = UserKNNScorer(k=30, min_nbrs=2)
    knnUU_pipe = topn_pipeline(knnUU_model, predicts_ratings=True)
    knnUU_pipe.train(split.train)

    knnUU_preds = batch.predict(knnUU_pipe, split.test)
    knnUU_recs = batch.recommend(knnUU_pipe, split.test, n=N)
    knnUU_recs.to_csv("knnUU_recommendations.csv", index=False)


    # Measure the recommendation quality (using RunAnalysis).
    ra1 = RunAnalysis()
    ra1.add_metric(NDCG(k=N))
    knnUU_results1 = ra1.compute(knnUU_recs, split.test)
    # print(knnUU_results1.list_summary())
    
    ra2 = RunAnalysis()
    ra2.add_metric(RMSE())
    knnUU_pred_results = ra2.compute(knnUU_preds, split.test)
    # print(knnUU_pred_results.list_summary())
    return knnUU_results1.list_summary(), knnUU_pred_results.list_summary()

def ItemItemKnnModelTrainer(split):
    knnII_model = ItemKNNScorer(k=20, min_nbrs=2, feedback='implicit')
    knnII_pipe = topn_pipeline(knnII_model, predicts_ratings=True)
    knnII_pipe.train(split.train)

    knnII_preds = batch.predict(knnII_pipe, split.test)
    knnII_recs = batch.recommend(knnII_pipe, split.test, n=N)
    knnII_recs.to_csv("knnII_recommendations.csv", index=False)


    # Measure the recommendation quality (using RunAnalysis).
    ra1 = RunAnalysis()
    ra1.add_metric(NDCG(k=N))
    knnII_results1 = ra1.compute(knnII_recs, split.test)
    # print(knnII_results1.list_summary())
    
    ra2 = RunAnalysis()
    ra2.add_metric(RMSE())
    knnII_pred_results = ra2.compute(knnII_preds, split.test)
    # print(knnII_pred_results.list_summary())
    return knnII_results1.list_summary(), knnII_pred_results.list_summary()

    

    