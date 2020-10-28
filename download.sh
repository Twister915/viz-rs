#!/bin/bash

set -e

fail() {
  error "${@}";
  exit 1;
}

error() {
  echo "error: ${@}" 1>&2
}

if [[ -z "${1}" ]]; then
  fail "please specify the source url";
fi;

if [[ -z "${2}" ]]; then
  fail "please specify the name to save the file as";
fi;

youtube-dl -o "${2}.%(ext)s" --audio-format wav -x "${1}"