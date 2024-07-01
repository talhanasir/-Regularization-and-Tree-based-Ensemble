library(keras)
library(keras)
library(tfruns)

args <- flags(
  flag_numeric("dropout1", 0.1),
  flag_numeric("dropout2", 0.2),
  flag_integer("batch_size", 32)
)

model <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = 'relu', input_shape = c(ncol(trainX))) %>%
  layer_dropout(rate = args$dropout1) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = args$dropout2) %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

history <- model %>% fit(
  features_train,
  trainResponse,
  epochs = 10,
  batch_size = args$batch_size,
  validation_data = list(features_val, valResponse)
  )