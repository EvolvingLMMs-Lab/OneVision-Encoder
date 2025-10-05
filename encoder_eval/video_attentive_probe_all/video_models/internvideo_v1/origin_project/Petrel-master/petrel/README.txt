Petrel
======

Tools for writing, submitting, debugging, and monitoring Storm topologies in pure Python.

Overview
========

Petrel offers some important improvements over the storm.py module provided with Storm:

* Topologies are implemented in 100% Python
* Petrel's packaging support automatically sets up a Python virtual environment for your topology and makes it easy to install additional Python packages.
* "petrel.mock" allows testing of single components or single chains of related components.
* Petrel automatically sets up logging for every spout or bolt and logs a stack trace on unhandled errors.