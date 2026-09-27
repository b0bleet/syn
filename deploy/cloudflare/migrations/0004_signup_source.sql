-- Where each account came from (a fixed channel name, see src/source.ts), and when it first
-- got an answer from the API. Both feed the aggregate signup funnel on /stats.
ALTER TABLE user ADD COLUMN signupSource TEXT;
ALTER TABLE user ADD COLUMN firstCallAt INTEGER;
