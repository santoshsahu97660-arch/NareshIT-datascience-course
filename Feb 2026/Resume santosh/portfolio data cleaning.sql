DROP DATABASE IF EXISTS world_layoffs;
CREATE DATABASE world_layoffs;
USE world_layoffs;

CREATE TABLE layoffs (
company VARCHAR(100),
location VARCHAR(100),
industry VARCHAR(100),
total_laid_off INT,
percentage_laid_off VARCHAR(20),
date VARCHAR(20),
stage VARCHAR(50),
country VARCHAR(100),
funds_raised_millions INT
);

INSERT INTO layoffs VALUES
('Amazon','Seattle','Retail',10000,'10%','1/5/2023','Post-IPO','United States.',108),
('Amazon','Seattle','Retail',10000,'10%','1/5/2023','Post-IPO','United States.',108),
('Google','California','Tech',12000,'6%','2/10/2023','Post-IPO','United States',200),
('Google','California','',12000,'6%','2/10/2023','Post-IPO','United States',200),
('Meta','California','Social Media',11000,'13%','11/9/2022','Post-IPO','United States',150),
('Meta','California','Social Media',11000,'13%','11/9/2022','Post-IPO','United States',150),
('Coinbase','Remote','Crypto Currency',2000,'18%','6/14/2022','Post-IPO','United States',547),
('Coinbase','Remote','CryptoCurrency',2000,'18%','6/14/2022','Post-IPO','United States',547),
('StartupX','Bangalore',NULL,NULL,NULL,'3/1/2023','Seed','India',5),
('StartupX','Bangalore',NULL,NULL,NULL,'3/1/2023','Seed','India',5);

CREATE TABLE layoffs_staging LIKE layoffs;
INSERT INTO layoffs_staging SELECT * FROM layoffs;

CREATE TABLE layoffs_staging2 (
company VARCHAR(100),
location VARCHAR(100),
industry VARCHAR(100),
total_laid_off INT,
percentage_laid_off VARCHAR(20),
date VARCHAR(20),
stage VARCHAR(50),
country VARCHAR(100),
funds_raised_millions INT,
row_num INT
);

INSERT INTO layoffs_staging2
SELECT *,
ROW_NUMBER() OVER (
PARTITION BY company, location, industry, total_laid_off,
percentage_laid_off, date, stage, country, funds_raised_millions
) AS row_num
FROM layoffs_staging;

SET SQL_SAFE_UPDATES = 0;
DELETE FROM layoffs_staging2 WHERE row_num >= 2;
SET SQL_SAFE_UPDATES = 1;

UPDATE layoffs_staging2
SET industry = NULL
WHERE industry = '';

UPDATE layoffs_staging2 t1
JOIN layoffs_staging2 t2
ON t1.company = t2.company
SET t1.industry = t2.industry
WHERE t1.industry IS NULL
AND t2.industry IS NOT NULL;

UPDATE layoffs_staging2
SET industry = 'Crypto'
WHERE industry IN ('Crypto Currency','CryptoCurrency');

UPDATE layoffs_staging2
SET country = TRIM(TRAILING '.' FROM country);

UPDATE layoffs_staging2
SET date = STR_TO_DATE(date,'%m/%d/%Y');

ALTER TABLE layoffs_staging2
MODIFY COLUMN date DATE;

-- remove useless rows
DELETE FROM layoffs_staging2
WHERE total_laid_off IS NULL
AND percentage_laid_off IS NULL;

ALTER TABLE layoffs_staging2
DROP COLUMN row_num;

-- final clean data
SELECT * FROM layoffs_staging2;
