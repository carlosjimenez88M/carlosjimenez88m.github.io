DROP TABLE IF EXISTS canciones;

CREATE TABLE canciones (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  artista TEXT NOT NULL,
  album TEXT NOT NULL,
  cancion TEXT NOT NULL,
  letra TEXT
);

CREATE UNIQUE INDEX idx_cancion_unica
ON canciones (artista, album, cancion);

INSERT INTO canciones (artista, album, cancion, letra) VALUES
  ('Pink Floyd', 'The Dark Side of the Moon', 'Speak to Me', NULL),
  ('Pink Floyd', 'The Dark Side of the Moon', 'Breathe (In the Air)', NULL),
  ('Pink Floyd', 'The Dark Side of the Moon', 'On the Run', NULL),
  ('Pink Floyd', 'The Dark Side of the Moon', 'Time', NULL),
  ('Pink Floyd', 'The Dark Side of the Moon', 'The Great Gig in the Sky', NULL),
  ('Pink Floyd', 'The Dark Side of the Moon', 'Money', NULL),
  ('Pink Floyd', 'The Dark Side of the Moon', 'Us and Them', NULL),
  ('Pink Floyd', 'The Dark Side of the Moon', 'Any Colour You Like', NULL),
  ('Pink Floyd', 'The Dark Side of the Moon', 'Brain Damage', NULL),
  ('Pink Floyd', 'The Dark Side of the Moon', 'Eclipse', NULL),
  ('The Beatles', 'Abbey Road', 'Come Together', NULL),
  ('The Beatles', 'Abbey Road', 'Something', NULL),
  ('The Beatles', 'Abbey Road', 'Maxwell''s Silver Hammer', NULL),
  ('The Beatles', 'Abbey Road', 'Oh! Darling', NULL),
  ('The Beatles', 'Abbey Road', 'Octopus''s Garden', NULL),
  ('The Beatles', 'Abbey Road', 'I Want You (She''s So Heavy)', NULL),
  ('The Beatles', 'Abbey Road', 'Here Comes the Sun', NULL),
  ('The Beatles', 'Abbey Road', 'Because', NULL),
  ('The Beatles', 'Abbey Road', 'You Never Give Me Your Money', NULL),
  ('The Beatles', 'Abbey Road', 'Sun King', NULL),
  ('The Beatles', 'Abbey Road', 'Mean Mr. Mustard', NULL),
  ('The Beatles', 'Abbey Road', 'Polythene Pam', NULL),
  ('The Beatles', 'Abbey Road', 'She Came In Through the Bathroom Window', NULL),
  ('The Beatles', 'Abbey Road', 'Golden Slumbers', NULL),
  ('The Beatles', 'Abbey Road', 'Carry That Weight', NULL),
  ('The Beatles', 'Abbey Road', 'The End', NULL),
  ('The Beatles', 'Abbey Road', 'Her Majesty', NULL);
