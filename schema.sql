CREATE TABLE properties (
    property_id SERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL,
    metro_area VARCHAR(255) NOT NULL,
    sq_footage INT NOT NULL,
    property_type VARCHAR(50) NOT NULL
);

CREATE TABLE financials (
    id SERIAL PRIMARY KEY,
    property_id INT NOT NULL REFERENCES properties(property_id) ON DELETE CASCADE,
    revenue NUMERIC(12, 2) NOT NULL,
    net_income NUMERIC(12, 2) NOT NULL,
    expenses NUMERIC(12, 2) NOT NULL
);