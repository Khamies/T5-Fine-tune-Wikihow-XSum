global_setting = {

"seed": 3111,

}


model_setting = {


"embed_size": 300,
"hidden_size": 256,
"latent_size": 16,
"note_size": 88,
"lstm_layer": 1

}


training_setting = {

"epochs": 1,
"tr_batch_size": 4,
"val_batch_size": 4,
"test_batch_size": 1,
"lr" : 0.001,
"metric": "rouge"

}