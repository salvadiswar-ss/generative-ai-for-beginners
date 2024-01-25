import os
import warnings

warnings.filterwarnings("ignore")

import copy
from pathlib import Path


import re
import sys
import time
import traceback
import pandas as pd
import google.cloud as cloud
from datetime import datetime, timedelta
from google.cloud import storage
from google.cloud import aiplatform
from io import StringIO

import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import pandas as pd
import torch

import pytorch_forecasting
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.timeseries import GroupNormalizer
from pytorch_forecasting.metrics import MAE,MAPE,SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.data.encoders import NaNLabelEncoder


# Download a table
def download_table(bq_table_uri: str):
    # Remove bq:// prefix if present
    prefix = "bq://"
    if bq_table_uri.startswith(prefix):
        bq_table_uri = bq_table_uri[len(prefix) :]
        
    # Download the BigQuery table as a dataframe
    # This requires the "BigQuery Read Session User" role on the custom training service account.
    table = bq_client.get_table(bq_table_uri)
    return bq_client.list_rows(table).to_dataframe()

def load_selected_features(data):
    '''
    Function used to load selected columns from BQ source table for training dataset
    '''
    features_selected  = ['ts_id','datetime','gross_quantity','business_area','cust_country','cust_region','customer_tier','funding_pct_msrp','product_area','product_family','product_line','promotion_type','region','sales_channel','weeks_since_launch','sellin_quantity']
    filtered_data = data[features_selected]
    return filtered_data

def data_preparation(data):
    data['datetime']= pd.to_datetime(data['datetime'], errors='coerce')
    data = data.dropna(subset=['datetime'])
    # add time index
    data["time_idx"] = (data["datetime"] - data["datetime"].min()).dt.days // 7
    max_prediction_length = 52
    max_encoder_length = 104
    training_cutoff = "2022-04-01 00:00:00"
    training_data= data[lambda x: x.datetime<=training_cutoff]
    print(training_data.shape)
    # print(training_data.tail(15))
    # validation_cutoff = data[data["time_idx"] > training_cutoff]["time_idx"].max()
    validation_data= data[lambda x: x.datetime>training_cutoff]
    print(validation_data.shape)

    training=TimeSeriesDataSet(
        training_data,
        time_idx="time_idx",
        target="gross_quantity",
        group_ids=["ts_id"],
        #allow_missing_timesteps=True,
        min_encoder_length= max_encoder_length//4,
        max_encoder_length=max_encoder_length,
        #min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["business_area","product_area", "product_family","cust_country",
                             "cust_region","customer_tier","region","sales_channel"],
        time_varying_known_categoricals=["promotion_type"],
        time_varying_unknown_categoricals=["product_line"],
        time_varying_known_reals=["time_idx",
                  "datetime",
                  "funding_pct_msrp",
                  "weeks_since_launch"],
        time_varying_unknown_reals=[
            "gross_quantity","sellin_quantity"
        ],
        target_normalizer= None,
        categorical_encoders={'product_family': NaNLabelEncoder(add_nan=True),
                             'product_line': NaNLabelEncoder(add_nan=True)
                              },
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )


    #print(dir(training))
    #params = training.get_parameters()
    #print(params)

    # create validation dataset using the same normalization techniques as for the training dataset
    validation = TimeSeriesDataSet.from_dataset(training,data,min_prediction_idx=training.index.time.max()+1,predict=True,stop_randomization=True)
    #validation = TimeSeriesDataSet.from_parameters(data=validation_data,parameters=training.get_parameters(), predict=True, stop_randomization=True)
    num_training_samples = len(training)
    num_validation_samples = len(validation)
    print(num_training_samples,num_validation_samples)
    # print(training.index.time.max())
    # print(validation.index.time.max())
    # # # # create dataloaders for model
    batch_size = 32  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
    print(len(train_dataloader))
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)
    print(len(val_dataloader))
    return training, train_dataloader, val_dataloader

def train_tft_network(train_dataloader, val_dataloader):
    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs") 

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        #enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=30,  # coment in for training, running valiation every 30 batches
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss= QuantileLoss(),
        #log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        optimizer="Ranger",
        reduce_on_plateau_patience=4,
    )

    trainer.fit(
    tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )


    # load the best model according to the validation loss
    # (given that we use early stopping, this is not necessarily the last epoch)
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(best_model_path)
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    #calcualte mean absolute error on validation set
    # predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
    # MAE()(predictions.output, predictions.y)

    return best_tft



def validation_plots(best_tft,val_dataloader):

    predictions, x = best_tft.predict(val_dataloader, return_x=True)
    predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
    all_features = list(set(predictions_vs_actuals['support'].keys())-set(['product_line']))
    for feature in all_features:
        best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals, name=feature);

if __name__ == '__main__':

    #### ML Pipeline for execution
    ### Training Data BQ Table
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bq_input_table, type=str, default=100, help="Input Data Path"
    )
    
    args = parser.parse_args()

    data= download_data(args.bq_input_table)
    data = load_selected_features(data)
    training, train_dataloader, val_dataloader = data_preparation(data)


    model = train_tft_network(train_dataloader, val_dataloader)
