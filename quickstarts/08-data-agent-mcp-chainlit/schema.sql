DROP TABLE IF EXISTS users;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    is_customer BOOLEAN DEFAULT FALSE,
    churn_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
