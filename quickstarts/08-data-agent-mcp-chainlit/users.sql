INSERT INTO users (name, email, is_customer, churn_reason, created_at) VALUES
('Alice Johnson', 'alice@example.com', TRUE, NULL, NOW() - INTERVAL '6 months'),
('Bob Smith', 'bob@example.com', TRUE, NULL, NOW() - INTERVAL '5 months'),
('Carla Ruiz', 'carla@example.com', TRUE, NULL, NOW() - INTERVAL '4 months'),
('David Lee', 'david@example.com', TRUE, NULL, NOW() - INTERVAL '3 months'),
('Emma Chen', 'emma@example.com', TRUE, NULL, NOW() - INTERVAL '2 months'),
('Frank Novak', 'frank@example.com', FALSE, 'Spent 10 minutes trying to find the logout button — it was hidden in a weird place.', NOW() - INTERVAL '6 weeks'),
('Grace Patel', 'grace@example.com', FALSE, 'The dashboard felt cluttered and overwhelming right from the start.', NOW() - INTERVAL '5 weeks'),
('Hassan Ali', 'hassan@example.com', FALSE, 'Couldn’t figure out how to edit my profile — had to Google it.', NOW() - INTERVAL '4 weeks'),
('Isabella Moreau', 'isabella@example.com', FALSE, 'Forms had way too many fields and no clear labels.', NOW() - INTERVAL '3 weeks'),
('Jamal Wright', 'jamal@example.com', FALSE, 'Nothing looked clickable — I was stuck on the home screen for a while.', NOW() - INTERVAL '2 weeks');
