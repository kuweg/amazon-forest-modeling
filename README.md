<<<<<<< README.md
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

## 2.Установка пакетов

В активированном окружении запустить команду
```
pip install -r requirements.txt
```

## 3.Загрузка датасета

https://www.dropbox.com/s/31zrqkkyl6vkxvz/planet.zip?dl=0


## 4.Настройка ClearML

Для начала работы с [ClearML](https://clear.ml/) потребуется создать аккаунт.

Переходим в свой профиль и находим "Create new credentials".

Далее в вводим команду: 

```
clearml-init
``` 

и следуем инструкциям.
За дополнительной информацией можно обратиться в этот [гайд](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps/)
