#!/bin/sh

mkdir buffer_folder                                                                 &&\
cd ./buffer_folder/                                                                 &&\
curl https://www.dropbox.com/s/31zrqkkyl6vkxvz/planet.zip?dl=0 -L -o data.zip       &&\
unzip data.zip                                                                      &&\
rm data.zip                                                                         &&\
rm planet/sample_submission.csv                                                     &&\
rm -rf planet/test-jpg                                                              &&\
cd ..                                                                               &&\                                                                        &&\
cp -R buffer_folder/planet/train_classes.csv data/                                 &&\
cp -R buffer_folder/planet/train-jpg data/                                         &&\
rm -r buffer_folder/                                                                   
