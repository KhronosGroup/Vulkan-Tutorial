# Copyright (c) 2026, Khronos Group and contributors
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#Function to clone or update one pinned Git checkout

# Check script arguments.
foreach(required_var IN ITEMS GIT_EXECUTABLE GIT_REPOSITORY GIT_TAG SOURCE_DIR)
  if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
    message(FATAL_ERROR "fetch_git.cmake requires ${required_var}")
  endif()
endforeach()

# Run one Git command and stop configuration immediately if it fails.
function(run_git)
  execute_process( COMMAND "${GIT_EXECUTABLE}" ${ARGV} COMMAND_ERROR_IS_FATAL ANY)
endfunction()

# SOURCE_DIR is the final checkout path requested by CMakeLists.txt.
# Create its parent directory before cloning into it.
get_filename_component(source_parent "${SOURCE_DIR}" DIRECTORY)
file(MAKE_DIRECTORY "${source_parent}")

set(created_repository OFF)
if(NOT EXISTS "${SOURCE_DIR}/.git")
  message( STATUS "GIT_FETCH: Creating repository ${SOURCE_DIR} from ${GIT_REPOSITORY}")
  set(clone_args clone --no-checkout)
  if(DEFINED SPARSE_PATH AND NOT "${SPARSE_PATH}" STREQUAL "")
    list(APPEND clone_args --filter=blob:none --sparse)
  endif()
  run_git(${clone_args} "${GIT_REPOSITORY}" "${SOURCE_DIR}")
  set(created_repository ON)
endif()

if(DEFINED SPARSE_PATH AND NOT "${SPARSE_PATH}" STREQUAL "")
  message( STATUS "GIT_FETCH: Setting sparse checkout for ${SOURCE_DIR}: ${SPARSE_PATH}")
  run_git(-C "${SOURCE_DIR}" sparse-checkout set "${SPARSE_PATH}")
endif()

# Read the current HEAD commit. A newly-created --no-checkout clone can already
# point HEAD at GIT_TAG, but still needs checkout to populate its working tree.
execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" rev-parse --verify HEAD
  OUTPUT_VARIABLE current_commit
  RESULT_VARIABLE head_result
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# If the checkout already points at the pinned commit, leave it untouched.
# If the repository is created it will be empty until checkout.
if(NOT created_repository AND current_commit STREQUAL GIT_TAG)
  message(STATUS "GIT_FETCH: Skipping ${SOURCE_DIR}; already at ${GIT_TAG}")
else()
  run_git(-C "${SOURCE_DIR}" fetch --filter=blob:none origin)
  message(STATUS "GIT_FETCH: Checking out ${SOURCE_DIR} at ${GIT_TAG}")
  run_git(-C "${SOURCE_DIR}" checkout --detach "${GIT_TAG}")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" rev-parse --verify HEAD
    OUTPUT_VARIABLE checked_out_commit
    COMMAND_ERROR_IS_FATAL ANY
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  message( STATUS "GIT_FETCH: Checked out ${SOURCE_DIR} at ${checked_out_commit}")
endif()
