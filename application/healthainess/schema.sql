DROP TABLE IF EXISTS exercises_done;

CREATE TABLE exercises_done
(
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT      NOT NULL,
    duration_s  INTEGER   NOT NULL,
    quality     INTEGER   NOT NULL,
    finished_at TIMESTAMP NOT NULL UNIQUE
);


DROP TABLE IF EXISTS exercises_plan;

CREATE TABLE exercises_plan
(
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL,
    duration_s INTEGER NOT NULL
);
