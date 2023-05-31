# Домашнее задание 1.1: Моделинг

## 1. Установка проекта и активация окружения
склонировать проект

```
https://gitlab.com/kuweg/modeling.git

cd modeling/
```

Создать и активировать виртуальное окружение.
Рекомендуется использовать `Python 3.9` (работа вполнялась на `Python 3.9.13`)

```
python -m venv <virtual_env_name>
```
```
Linux: 
    source /path/to/new/virtual/env/bin/activate

Windows:
    /path/to/new/virtual/env/Source/activate
```

## 2. Установка пакетов

В активированном окружении запустить команду
```
pip install -r requirements.txt
```

## 3. Загрузка датасета

Для загрузки датасета выполнить команду:

```
python get_data.py path/to/config.yaml [Optional] --rewrite
```
где, `rewrite` - перезаписать директорию или нет.

Если что-то пошло не так:


AmazonForest датасет загружен на гугл диск, могло что-то случиться с ссылкой.

Для Linux:
```
./fetch_data.bash
```

Для Windows:

1. Скачайте датасет по [ссылке]('https://drive.google.com/file/d/10SQ1bXpkqVgqE9_g3_zl9shFVZ91QGEV/view?usp=drive_link')
2. В архиве будет папка `planet`, из этой папки перенесите все в ту папку, что прописана в 
        `congif.data_config.data_path`
3. *Можно удалить `/test-jpg` и `sample_submission.csv`, так как они не используются

Папка `data` в итоге должна иметь такую структуру:
```
data\
    train-jpg\
    train_classes.csv
```

## 4. Настройка ClearML

Для начала работы с [ClearML](https://clear.ml/) потребуется создать аккаунт.

Переходим в свой профиль и находим "Create new credentials".

Далее в вводим команду: 

```
clearml-init
``` 

и следуем инструкциям.
За дополнительной информацией можно обратиться в этот [гайд](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps/)

## 5. Запуск обучения
Для запуска обучения:
```
python train.py configs/config.yaml
```

