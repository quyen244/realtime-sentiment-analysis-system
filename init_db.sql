
create DATABASE absa_db;

CREATE TABLE sentiment_analysis(
    id SERIAL PRIMARY KEY,
    review TEXT NOT NULL,
    pred_price INTEGER,
    pred_shipping INTEGER,
    pred_outlook INTEGER,
    pred_quality INTEGER,
    pred_size INTEGER,
    pred_shop_service INTEGER,
    pred_general INTEGER,
    pred_others INTEGER,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE model_versions(
     version varchar(50) NOT NULL PRIMARY KEY,
    model_path text NOT NULL,
    accuracy_price double precision,
    accuracy_shipping double precision,
    accuracy_outlook double precision,
    accuracy_quality double precision,
    accuracy_size double precision,
    accuracy_shop_service double precision,
    accuracy_general double precision,
    accuracy_others double precision,
    avg_accuracy double precision,
    f1_score_price double precision,
    f1_score_shipping double precision,
    f1_score_outlook double precision,
    f1_score_quality double precision,
    f1_score_size double precision,
    f1_score_shop_service double precision,
    f1_score_general double precision,
    f1_score_others double precision,
    avg_f1_score double precision,
    is_production boolean DEFAULT false,
    notes text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
);
