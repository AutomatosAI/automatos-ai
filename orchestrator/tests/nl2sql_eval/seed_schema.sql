-- PRD-160 S5 — seeded schema for the NL2SQL regression eval.
-- Portable SQL (runs on SQLite in-memory AND Postgres) so the eval needs no
-- external service. Deterministic data → deterministic golden result sets.

CREATE TABLE customers (
    id         INTEGER PRIMARY KEY,
    name       VARCHAR(100) NOT NULL,
    country    VARCHAR(50)  NOT NULL,
    status     VARCHAR(20)  NOT NULL,
    created_at VARCHAR(10)  NOT NULL
);

CREATE TABLE products (
    id       INTEGER PRIMARY KEY,
    name     VARCHAR(100) NOT NULL,
    category VARCHAR(50)  NOT NULL,
    price    DECIMAL(10,2) NOT NULL
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date  VARCHAR(10) NOT NULL,
    total       DECIMAL(10,2) NOT NULL,
    status      VARCHAR(20) NOT NULL
);

CREATE TABLE order_items (
    id         INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity   INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL
);

INSERT INTO customers (id, name, country, status, created_at) VALUES
    (1, 'Acme Corp',     'USA',     'active',   '2025-01-05'),
    (2, 'Globex',        'USA',     'active',   '2025-02-11'),
    (3, 'Initech',       'UK',      'churned',  '2025-01-20'),
    (4, 'Umbrella',      'Germany', 'active',   '2025-03-02'),
    (5, 'Soylent',       'USA',     'active',   '2025-03-15'),
    (6, 'Hooli',         'UK',      'active',   '2025-04-01'),
    (7, 'Vehement',      'France',  'churned',  '2025-02-28'),
    (8, 'Massive Dynamic','USA',    'active',   '2025-04-10');

INSERT INTO products (id, name, category, price) VALUES
    (1, 'Widget',     'Hardware',    9.99),
    (2, 'Gadget',     'Hardware',   19.99),
    (3, 'Dashboard',  'Software',   49.00),
    (4, 'Analytics',  'Software',   99.00),
    (5, 'Cable',      'Accessory',   4.50),
    (6, 'Adapter',    'Accessory',  12.00),
    (7, 'Server',     'Hardware',  499.00),
    (8, 'License',    'Software',  299.00);

INSERT INTO orders (id, customer_id, order_date, total, status) VALUES
    (1, 1, '2025-04-02', 119.98, 'completed'),
    (2, 1, '2025-04-20',  49.00, 'completed'),
    (3, 2, '2025-04-22', 598.00, 'completed'),
    (4, 3, '2025-03-30',  19.99, 'cancelled'),
    (5, 4, '2025-04-25', 299.00, 'completed'),
    (6, 5, '2025-05-01',  16.50, 'pending'),
    (7, 6, '2025-05-03',  99.00, 'completed'),
    (8, 2, '2025-05-05', 499.00, 'pending'),
    (9, 8, '2025-05-06', 108.99, 'completed'),
    (10,1, '2025-05-07',   9.99, 'cancelled');

INSERT INTO order_items (id, order_id, product_id, quantity, unit_price) VALUES
    (1, 1, 1, 2,  9.99),
    (2, 1, 4, 1, 99.00),
    (3, 2, 3, 1, 49.00),
    (4, 3, 7, 1,499.00),
    (5, 3, 4, 1, 99.00),
    (6, 4, 2, 1, 19.99),
    (7, 5, 8, 1,299.00),
    (8, 6, 6, 1, 12.00),
    (9, 6, 5, 1,  4.50),
    (10,7, 4, 1, 99.00),
    (11,8, 7, 1,499.00),
    (12,9, 3, 1, 49.00),
    (13,9, 4, 1, 99.00),
    (14,10,1, 1,  9.99);
