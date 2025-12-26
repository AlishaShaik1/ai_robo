import pkg_resources
with open("installed_packages.txt", "w") as f:
    for dist in pkg_resources.working_set:
        f.write(f"{dist.project_name}=={dist.version}\n")
