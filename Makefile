run:
\tuvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
\tpytest -q

lint:
\truff .

format:
\tblack .
