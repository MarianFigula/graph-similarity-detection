todo:
- pridat vahy pre metody labelovania


## Supervised learning

pre trenovanie s ucitelom, modely:

- MLP
- Random Forest Classifier

Vybrat: similarity s labelami .csv - input entries s buttonom

hyperparameters:

RFC:
- n_estimators - ok
- max_depth - ok
- min_samples_split - ok
- min_samples_leaf - ok
- batch_size - ok 
- max_features - ok

plotting:
 - plot decision tree by index (mozno)
 - plot x most important features
 - plot confusion matrix
 - roc curve
 - precision recall f1 curve

MLP:
- batch_size - ok
- learning_rate - OK
- num_epochs - ok
- num_hidden_layers - ok
- neurons_per_layer - partial done
- dropout_rate - partial done
- early_stopping (true/false) - ak early_stopping je true tak set patience


plotting:
 - accuracy/loss curve - ok
 - confusion matrix - ok
 - roc curve - ok
 - precision recall f1 curve

## Unsupervised learning

Vybrat: graphlet counts .csv - input entry s buttonom


### Moznost natrenovat autoencoder
hyperparameters:

- batch_size
- learning_rate
- num_epochs
- input_dim - 30 - asi tam nebude
- encoding_dim
- output_dim - preddefinovana 5
- cosine/euclidean - preddefinovana cosine


### Moznost natrenovat kmeans
hyperparameters:
- num_clusters - preddefinovane 4

### Moznost natrenovat DBSCAN
hyperparameters:
- eps - preddefinovane 0.1
- min_samples - preddefinovane 10