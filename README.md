# Домашнее задание 1.1: Моделинг

## 1. Установка проекта и активация окружения
Склонировать проект

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

## 2.Установка пакетов

В активированном окружении запустить команду
```
pip install -r requirements.txt
```

## 3.Загрузка датасета

Выполнить эту команду:

```
python get_data.py <path/to/config.yaml> [Optinal] --rewrite
```
где `rewrite` - булевой флаг перезаписывать директорию или нет.

Если все прошло успешно, то директория даты примет такой вид:
```
data\
    train-jpg\
    train_classes.csv
```

Файл загружается через небольшой костыль от гугл диска.
Если что-то пошло не так, то я подготовил еще пару:

Для Linux:
```
./fetch_data.bash
```

Для Windows:
1. Скачать архив по [ссылке](https://drive.google.com/file/d/10SQ1bXpkqVgqE9_g3_zl9shFVZ91QGEV/view?usp=drive_link)
2. Распаковать в любое удобное место
3. Из папки `planet` пренести `train-jpg\` и `train_classes.csv`


## 4.Настройка ClearML

Для начала работы с [ClearML](https://clear.ml/) потребуется создать аккаунт.

Переходим в свой профиль и находим "Create new credentials".

Далее в вводим команду: 

```
clearml-init
``` 

и следуем инструкциям.
За дополнительной информацией можно обратиться в этот [гайд](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps/)


## 5. Запуск обучения

Запуск тренировки:
```
python train.py <path/to/config.yaml>
```

## 6. Инференс

После обучения можно посмотреть как работает инференс модели в `notebooks/inference.ipynb`


Чтобы получить мой лучший эксперимент:
```
dvc pull
```
Также его [лог](https://app.clear.ml/projects/16c462cff38f4020a6032b9eed8f3da8/experiments/6d78bdbf0d5d45ecbb7d2ecaa82cf366/output/execution) в ClearML (нужен аккаунт)

Касательно `.ipynb` файлов и редакторов кода. Я пользуюсь `VSCODE`, который имеет сносную поддержку тетрадок и попросит что-нибудь доустановить без вреда для здоровья, однако иногда он может сойти с ума и сломаться.


 Поэтому лучше будет воспользоваться `Jupyter Lab` или `Jupyter Notebook` для надежности.

 ## 7. Тесты

 Для запуска тестов выполнить

 ```
 python -m pytest .\tests\unit\ -v
 ```