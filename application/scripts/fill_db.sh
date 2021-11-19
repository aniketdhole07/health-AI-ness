#!/bin/bash

curl --location --request GET 'http://0.0.0.0:5000//api/exercises_done/deleteAll'


curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 3",
    "duration_s": 65,
    "quality": 99,
    "finished_at": "2021-09-22 08:01:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 3",
    "duration_s": 40,
    "quality": 99,
    "finished_at": "2021-09-22 08:03:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 3",
    "duration_s": 30,
    "quality": 59,
    "finished_at": "2021-09-22 08:07:02"
}'


curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 1",
    "duration_s": 60,
    "quality": 99,
    "finished_at": "2021-09-22 09:01:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 1",
    "duration_s": 60,
    "quality": 99,
    "finished_at": "2021-09-22 09:04:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 2",
    "duration_s": 110,
    "quality": 90,
    "finished_at": "2021-09-22 09:08:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 2",
    "duration_s": 11,
    "quality": 55,
    "finished_at": "2021-09-22 09:13:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 3",
    "duration_s": 70,
    "quality": 99,
    "finished_at": "2021-09-23 08:03:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 3",
    "duration_s": 60,
    "quality": 59,
    "finished_at": "2021-09-23 08:07:02"
}'


curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 1",
    "duration_s": 70,
    "quality": 99,
    "finished_at": "2021-09-23 09:01:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 1",
    "duration_s": 50,
    "quality": 99,
    "finished_at": "2021-09-23 09:04:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 2",
    "duration_s": 120,
    "quality": 90,
    "finished_at": "2021-09-23 09:08:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 2",
    "duration_s": 120,
    "quality": 55,
    "finished_at": "2021-09-23 09:13:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 3",
    "duration_s": 40,
    "quality": 99,
    "finished_at": "2021-09-24 08:03:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 3",
    "duration_s": 30,
    "quality": 59,
    "finished_at": "2021-09-24 08:07:02"
}'


curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 1",
    "duration_s": 70,
    "quality": 99,
    "finished_at": "2021-09-24 09:01:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 1",
    "duration_s": 70,
    "quality": 99,
    "finished_at": "2021-09-24 09:04:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 2",
    "duration_s": 140,
    "quality": 90,
    "finished_at": "2021-09-24 09:08:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 2",
    "duration_s": 130,
    "quality": 55,
    "finished_at": "2021-09-24 09:13:02"
}'


curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 3",
    "duration_s": 60,
    "quality": 99,
    "finished_at": "2021-09-25 09:01:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 3",
    "duration_s": 59,
    "quality": 99,
    "finished_at": "2021-09-25 09:03:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 3",
    "duration_s": 30,
    "quality": 59,
    "finished_at": "2021-09-25 09:07:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 1",
    "duration_s": 160,
    "quality": 8,
    "finished_at": "2021-09-25 10:01:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 1",
    "duration_s": 180,
    "quality": 90,
    "finished_at": "2021-09-25 10:03:02"
}'

curl --location --request POST 'http://0.0.0.0:5000/api/exercises_done/create' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Exercise 1",
    "duration_s": 80,
    "quality": 59,
    "finished_at": "2021-09-25 10:07:02"
}'

