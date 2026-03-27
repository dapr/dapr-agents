CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    plan VARCHAR(50) NOT NULL DEFAULT 'free',
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    churned_at TIMESTAMP,
    signup_date DATE NOT NULL DEFAULT CURRENT_DATE,
    last_login TIMESTAMP,
    country VARCHAR(60),
    product_area VARCHAR(80)
);
