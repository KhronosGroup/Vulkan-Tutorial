# Copyright (c) 2023 Sascha Willems
# SPDX-License-Identifier: CC-BY-SA-4.0

# Does automated changes to the asciidoc files required to make them properly work with Antora

import os

for root, dirs, files in os.walk("modules/ROOT/pages/"):
    for file in files:
        if (file.endswith(".adoc")):    
            file_name = os.path.join(root, file);            
            print("Fixing " + file_name)
            s = ""
            with open(file_name, "r+") as f:
                if file in ["index.adoc"]:
                    continue
                s = f.read()
                f.seek(0)
                f.truncate()
                # We need to change the attachment links (e.g. used to link to the chapter's source files)
                # It's not possible to format them in a way to works both with github and Antora
                s = s.replace("link:/attachments/", "link:{attachmentsdir}/")
                f.write(s)
                f.close