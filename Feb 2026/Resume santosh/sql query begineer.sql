 DROP DATABASE IF EXISTS `Parks_and_Recreation`;
CREATE DATABASE `Parks_and_Recreation`;
USE `Parks_and_Recreation`;






CREATE TABLE employee_demographics (
  employee_id INT NOT NULL,
  first_name VARCHAR(50),
  last_name VARCHAR(50),
  age INT,
  gender VARCHAR(10),
  birth_date DATE,
  PRIMARY KEY (employee_id)
);

CREATE TABLE employee_salary (
  employee_id INT NOT NULL,
  first_name VARCHAR(50) NOT NULL,
  last_name VARCHAR(50) NOT NULL,
  occupation VARCHAR(50),
  salary INT,
  dept_id INT
);


INSERT INTO employee_demographics (employee_id, first_name, last_name, age, gender, birth_date)
VALUES
(1,'Leslie', 'Knope', 44, 'Female','1979-09-25'),
(3,'Tom', 'Haverford', 36, 'Male', '1987-03-04'),
(4, 'April', 'Ludgate', 29, 'Female', '1994-03-27'),
(5, 'Jerry', 'Gergich', 61, 'Male', '1962-08-28'),
(6, 'Donna', 'Meagle', 46, 'Female', '1977-07-30'),
(7, 'Ann', 'Perkins', 35, 'Female', '1988-12-01'),
(8, 'Chris', 'Traeger', 43, 'Male', '1980-11-11'),
(9, 'Ben', 'Wyatt', 38, 'Male', '1985-07-26'),
(10, 'Andy', 'Dwyer', 34, 'Male', '1989-03-25'),
(11, 'Mark', 'Brendanawicz', 40, 'Male', '1983-06-14'),
(12, 'Craig', 'Middlebrooks', 37, 'Male', '1986-07-27');


INSERT INTO employee_salary (employee_id, first_name, last_name, occupation, salary, dept_id)
VALUES
(1, 'Leslie', 'Knope', 'Deputy Director of Parks and Recreation', 75000,1),
(2, 'Ron', 'Swanson', 'Director of Parks and Recreation', 70000,1),
(3, 'Tom', 'Haverford', 'Entrepreneur', 50000,1),
(4, 'April', 'Ludgate', 'Assistant to the Director of Parks and Recreation', 25000,1),
(5, 'Jerry', 'Gergich', 'Office Manager', 50000,1),
(6, 'Donna', 'Meagle', 'Office Manager', 60000,1),
(7, 'Ann', 'Perkins', 'Nurse', 55000,4),
(8, 'Chris', 'Traeger', 'City Manager', 90000,3),
(9, 'Ben', 'Wyatt', 'State Auditor', 70000,6),
(10, 'Andy', 'Dwyer', 'Shoe Shiner and Musician', 20000, NULL),
(11, 'Mark', 'Brendanawicz', 'City Planner', 57000, 3),
(12, 'Craig', 'Middlebrooks', 'Parks Director', 65000,1);



CREATE TABLE parks_departments (
  department_id INT NOT NULL AUTO_INCREMENT,
  department_name varchar(50) NOT NULL,
  PRIMARY KEY (department_id)
);

INSERT INTO parks_departments (department_name)
VALUES
('Parks and Recreation'),
('Animal Control'),
('Public Works'),
('Healthcare'),
('Library'),
('Finance');

SELECT * 
FROM parks_and_recreation.employee_demographics; 

SELECT first_name
FROM employee_demographics; 

SELECT first_name, last_name
FROM employee_demographics; 

SELECT last_name, first_name, gender, age
FROM employee_demographics;

SELECT first_name, 
last_name,
salary,
salary + 100
FROM employee_salary;

SELECT first_name, 
last_name,
salary,
(salary + 100) * 10
FROM employee_salary;

SELECT *
FROM employee_salary
WHERE salary > 50000;

SELECT *
FROM employee_salary
WHERE salary >= 50000;

SELECT *
FROM employee_demographics
WHERE gender = 'Female';

SELECT *
FROM employee_demographics
WHERE gender != 'Female';

SELECT *
FROM employee_demographics
WHERE birth_date > '1985-01-01';

SELECT *
FROM employee_demographics
WHERE first_name LIKE 'a%';

SELECT *
FROM employee_demographics
WHERE first_name LIKE 'a__';

SELECT *
FROM employee_demographics
WHERE first_name LIKE 'a___%';


SELECT *
FROM employee_demographics;

SELECT gender
FROM employee_demographics
GROUP BY gender
;

SELECT occupation
FROM employee_salary
GROUP BY occupation
;
SELECT occupation, salary
FROM employee_salary
GROUP BY occupation, salary
;

SELECT gender, AVG(age)
FROM employee_demographics
GROUP BY gender
;

SELECT gender, MIN(age), MAX(age), COUNT(age),AVG(age)
FROM employee_demographics
GROUP BY gender
;

SELECT *
FROM customers
ORDER BY first_name;

SELECT *
FROM employee_demographics;

SELECT *
FROM employee_demographics
ORDER BY first_name;

SELECT *
FROM employee_demographics
ORDER BY first_name DESC;

SELECT *
FROM employee_demographics
ORDER BY gender, age;

SELECT *
FROM employee_demographics
ORDER BY gender DESC, age DESC;

SELECT *
FROM employee_demographics
ORDER BY 5 DESC, 4 DESC;


SELECT gender, AVG(age)
FROM employee_demographics
GROUP BY gender
HAVING AVG(age) > 40
;

SELECT gender, AVG(age) as AVG_age
FROM employee_demographics
GROUP BY gender
HAVING AVG_age > 40
;


SELECT *
FROM employee_demographics
LIMIT 3;

SELECT *
FROM employee_demographics
ORDER BY first_name
LIMIT 3;

SELECT *
FROM employee_demographics
ORDER BY first_name;

SELECT *
FROM employee_demographics
ORDER BY first_name
LIMIT 3,2;

SELECT *
FROM employee_demographics
ORDER BY age desc;

SELECT *
FROM employee_demographics
ORDER BY age desc
LIMIT 2,1;

SELECT gender, AVG(age)
FROM employee_demographics
GROUP BY gender
;
SELECT gender, AVG(age) AS Avg_age
FROM employee_demographics
GROUP BY gender
;
SELECT gender, AVG(age) Avg_age
FROM employee_demographics
GROUP BY gender
;
SELECT *
FROM employee_demographics;

SELECT *
FROM employee_salary;

SELECT *
FROM employee_demographics
JOIN employee_salary
	ON employee_demographics.employee_id = employee_salary.employee_id;
    
  SELECT *
FROM employee_demographics dem
INNER JOIN employee_salary sal
	ON dem.employee_id = sal.employee_id;  
SELECT *
FROM employee_salary sal
LEFT JOIN employee_demographics dem
	ON dem.employee_id = sal.employee_id;    
    
   SELECT *
FROM employee_salary sal
RIGHT JOIN employee_demographics dem
	ON dem.employee_id = sal.employee_id; 
    
  SELECT *
FROM employee_salary;  

SELECT *
FROM employee_salary emp1
JOIN employee_salary emp2
	ON emp1.employee_id = emp2.employee_id
    ;
   
  SELECT *
FROM employee_salary emp1
JOIN employee_salary emp2
	ON emp1.employee_id + 1  = emp2.employee_id
    ; 
    
 SELECT emp1.employee_id as emp_santa, emp1.first_name as santa_first_name, emp1.last_name as santa_last_name, emp2.employee_id, emp2.first_name, emp2.last_name
FROM employee_salary emp1
JOIN employee_salary emp2
	ON emp1.employee_id + 1  = emp2.employee_id
    ;   
    
 SELECT * 
FROM parks_and_recreation.parks_departments;

SELECT *
FROM employee_demographics dem
INNER JOIN employee_salary sal
	ON dem.employee_id = sal.employee_id
JOIN parks_departments dept
	ON dept.department_id = sal.dept_id;
   
   SELECT *
FROM employee_demographics dem
INNER JOIN employee_salary sal
	ON dem.employee_id = sal.employee_id
LEFT JOIN parks_departments dept
	ON dept.department_id = sal.dept_id; 
    
   SELECT first_name, last_name
FROM employee_demographics
UNION
SELECT occupation, salary
FROM employee_salary; 


SELECT first_name, last_name
FROM employee_demographics
UNION
SELECT first_name, last_name
FROM employee_salary; 

SELECT first_name, last_name
FROM employee_demographics
UNION DISTINCT
SELECT first_name, last_name
FROM employee_salary;

SELECT first_name, last_name
FROM employee_demographics
UNION ALL
SELECT first_name, last_name
FROM employee_salary;

SELECT first_name, last_name, 'Old'
FROM employee_demographics
WHERE age > 50;

SELECT first_name, last_name, 'Old Lady' as Label
FROM employee_demographics
WHERE age > 40 AND gender = 'Female'
UNION
SELECT first_name, last_name, 'Old Man'
FROM employee_demographics
WHERE age > 40 AND gender = 'Male'
UNION
SELECT first_name, last_name, 'Highly Paid Employee'
FROM employee_salary
WHERE salary >= 70000
ORDER BY first_name
;
SELECT LENGTH('sky');

SELECT first_name, LENGTH(first_name) 
FROM employee_demographics;

SELECT UPPER('sky');

SELECT first_name, UPPER(first_name) 
FROM employee_demographics;

SELECT LOWER('sky');

SELECT first_name, LOWER(first_name) 
FROM employee_demographics;

SELECT TRIM('sky'   );
SELECT LTRIM('     I           love          SQL');

SELECT LTRIM('     I love SQL');

SELECT RTRIM('I love SQL    ');

SELECT LEFT('Alexander', 4);
SELECT first_name, LEFT(first_name,4) 
FROM employee_demographics;
SELECT RIGHT('Alexander', 6);

SELECT first_name, RIGHT(first_name,4) 
FROM employee_demographics;

SELECT SUBSTRING('Alexander', 2, 3);

SELECT birth_date, SUBSTRING(birth_date,1,4) as birth_year
FROM employee_demographics;

SELECT REPLACE(first_name,'a','z')
FROM employee_demographics;

SELECT LOCATE('x', 'Alexander');

SELECT LOCATE('e', 'Alexander');

SELECT first_name, LOCATE('a',first_name) 
FROM employee_demographics;

SELECT first_name, LOCATE('Mic',first_name) 
FROM employee_demographics;

SELECT CONCAT('Alex', 'Freberg');

SELECT CONCAT(first_name, ' ', last_name) AS full_name
FROM employee_demographics;

SELECT * 
FROM employee_demographics;

SELECT first_name, 
last_name, 
CASE
	WHEN age <= 30 THEN 'Young'
END
FROM employee_demographics;

SELECT first_name, 
last_name, 
CASE
	WHEN age <= 30 THEN 'Young'
    WHEN age BETWEEN 31 AND 50 THEN 'Old'
    WHEN age >= 50 THEN "On Death's Door"
END
FROM employee_demographics;

SELECT * 
FROM employee_salary;

SELECT first_name, last_name, salary,
CASE
	WHEN salary > 45000 THEN salary + (salary * 0.05)
    WHEN salary < 45000 THEN salary + (salary * 0.07)
END AS new_salary
FROM employee_salary;

SELECT first_name, last_name, salary,
CASE
	WHEN salary > 45000 THEN salary + (salary * 0.05)
    WHEN salary < 45000 THEN salary + (salary * 0.07)
END AS new_salary,
CASE
	WHEN dept_id = 6 THEN salary * .10
END AS Bonus
FROM employee_salary;

SELECT *
FROM employee_demographics;

SELECT *
FROM employee_demographics
WHERE employee_id IN 
			(SELECT employee_id
				FROM employee_salary
                WHERE dept_id = 1);

SELECT *
FROM employee_demographics
WHERE employee_id IN 
			(SELECT employee_id, salary
				FROM employee_salary
                WHERE dept_id = 1);
                
 SELECT first_name, salary, AVG(salary)
FROM employee_salary
GROUP BY first_name, salary;

SELECT first_name, 
salary, 
(SELECT AVG(salary) 
	FROM employee_salary)
FROM employee_salary;

SELECT gender, AVG(Min_age)
FROM (SELECT gender, MIN(age) Min_age, MAX(age) Max_age, COUNT(age) Count_age ,AVG(age) Avg_age
FROM employee_demographics
GROUP BY gender) AS Agg_Table
GROUP BY gender
;

SELECT * 
FROM employee_demographics;

SELECT gender, ROUND(AVG(salary),1)
FROM employee_demographics dem
JOIN employee_salary sal
	ON dem.employee_id = sal.employee_id
GROUP BY gender
;

SELECT dem.employee_id, dem.first_name, gender, salary,
AVG(salary) OVER()
FROM employee_demographics dem
JOIN employee_salary sal
	ON dem.employee_id = sal.employee_id
;

SELECT dem.employee_id, dem.first_name, gender, salary,
AVG(salary) OVER(PARTITION BY gender)
FROM employee_demographics dem
JOIN employee_salary sal
	ON dem.employee_id = sal.employee_id
;

SELECT dem.employee_id, dem.first_name, gender, salary,
SUM(salary) OVER(PARTITION BY gender ORDER BY employee_id)
FROM employee_demographics dem
JOIN employee_salary sal
	ON dem.employee_id = sal.employee_id
;

SELECT dem.employee_id, dem.first_name, gender, salary,
ROW_NUMBER() OVER(PARTITION BY gender)
FROM employee_demographics dem
JOIN employee_salary sal
	ON dem.employee_id = sal.employee_id
;

SELECT dem.employee_id, dem.first_name, gender, salary,
ROW_NUMBER() OVER(PARTITION BY gender ORDER BY salary desc)
FROM employee_demographics dem
JOIN employee_salary sal
	ON dem.employee_id = sal.employee_id
;

SELECT dem.employee_id, dem.first_name, gender, salary,
ROW_NUMBER() OVER(PARTITION BY gender ORDER BY salary desc) row_num,
Rank() OVER(PARTITION BY gender ORDER BY salary desc) rank_1 
FROM employee_demographics dem
JOIN employee_salary sal
	ON dem.employee_id = sal.employee_id
;

SELECT dem.employee_id, dem.first_name, gender, salary,
ROW_NUMBER() OVER(PARTITION BY gender ORDER BY salary desc) row_num,
Rank() OVER(PARTITION BY gender ORDER BY salary desc) rank_1,
dense_rank() OVER(PARTITION BY gender ORDER BY salary desc) dense_rank_2 -- this is numerically ordered instead of positional like rank
FROM employee_demographics dem
JOIN employee_salary sal
	ON dem.employee_id = sal.employee_id
;

(
SELECT gender, SUM(salary), MIN(salary), MAX(salary), COUNT(salary), AVG(salary)
FROM employee_demographics dem
JOIN employee_salary sal
	ON dem.employee_id = sal.employee_id
GROUP BY gender
)

(
SELECT gender, SUM(salary), MIN(salary), MAX(salary), COUNT(salary)
FROM employee_demographics dem
JOIN employee_salary sal
	ON dem.employee_id = sal.employee_id
GROUP BY gender
)

(
SELECT employee_id, gender, birth_date
FROM employee_demographics dem
WHERE birth_date > '1985-01-01'
), 
(
SELECT employee_id, salary
FROM parks_and_recreation.employee_salary
WHERE salary >= 50000
)

(
SELECT gender, SUM(salary), MIN(salary), MAX(salary), COUNT(salary)
FROM employee_demographics dem
JOIN employee_salary sal
	ON dem.employee_id = sal.employee_id
GROUP BY gender
)

CREATE TEMPORARY TABLE temp_table
(first_name varchar(50),
last_name varchar(50),
favorite_movie varchar(100)
);

SELECT *
FROM temp_table;

INSERT INTO temp_table
VALUES ('Alex','Freberg','Lord of the Rings: The Twin Towers');

SELECT *
FROM temp_table;

CREATE TEMPORARY TABLE salary_over_50k
SELECT *
FROM employee_salary
WHERE salary > 50000;

CREATE PROCEDURE large_salaries()
SELECT *
FROM employee_salary
WHERE salary >= 60000;

CALL large_salaries();

CREATE PROCEDURE large_salaries2()
SELECT *
FROM employee_salary
WHERE salary >= 60000;
SELECT *
FROM employee_salary
WHERE salary >= 50000;

DELIMITER ;

USE `parks_and_recreation`;
DROP procedure IF EXISTS `large_salaries3`;

DELIMITER $$
CREATE PROCEDURE large_salaries3()
BEGIN
	SELECT *
	FROM employee_salary
	WHERE salary >= 60000;
	SELECT *
	FROM employee_salary
	WHERE salary >= 50000;
END $$

DELIMITER ;

USE `parks_and_recreation`;
DROP procedure IF EXISTS `large_salaries3`;
-- it automatically adds the dilimiter for us
DELIMITER $$
CREATE PROCEDURE large_salaries3(employee_id_param INT)
BEGIN
	SELECT *
	FROM employee_salary
	WHERE salary >= 60000
    AND employee_id_param = employee_id;
END $$

DELIMITER ;

CALL large_salaries3(1);

SELECT * FROM employee_salary;
SELECT * FROM employee_demographics;

USE parks_and_recreation;

DELIMITER $$

CREATE TRIGGER employee_insert2
AFTER INSERT ON employee_salary
 FOR EACH ROW
 BEGIN
 INSERT INTO employee_demographics (employee_id, first_name, last_name) VALUES (NEW.employee_id,NEW.first_name,NEW.last_name);
END $$

DELIMITER ; 

INSERT INTO employee_salary (employee_id, first_name, last_name, occupation, salary, dept_id)
VALUES(13, 'Jean-Ralphio', 'Saperstein', 'Entertainment 720 CEO', 1000000, NULL);

SELECT * 
FROM parks_and_recreation.employee_demographics;

SHOW EVENTS;

DROP EVENT IF EXISTS delete_retirees;
DELIMITER $$
CREATE EVENT delete_retirees
ON SCHEDULE EVERY 30 SECOND
DO BEGIN
	DELETE
	FROM parks_and_recreation.employee_demographics
    WHERE age >= 60;
END $$

SELECT * 
FROM parks_and_recreation.employee_demographics;
