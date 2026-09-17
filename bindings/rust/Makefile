VERSION=$(shell cat ../../VERSION)

deps: Cargo.toml sqlite-vec.c sqlite-vec.h sqlite3ext.h sqlite3.h

Cargo.toml: ../../VERSION Cargo.toml.tmpl
		VERSION=$(VERSION) envsubst < Cargo.toml.tmpl > $@

sqlite-vec.c: ../../sqlite-vec.c $(wildcard ../../sqlite-vec-*.c) ../../scripts/amalgamate.py
		uv run --no-project python ../../scripts/amalgamate.py $< > $@

sqlite-vec.h: ../../sqlite-vec.h
		cp $< $@

sqlite3ext.h: ../../vendor/sqlite3ext.h
		cp $< $@

sqlite3.h: ../../vendor/sqlite3.h
		cp $< $@

.PHONY: deps
