To configure Sublime Text 3 as a Python IDLE
By Romy Bompart

###Usage

1. First install Sublime Text 3
 1.1. Download the installator from : https://www.sublimetext.com/3

2. Once it is install you need to install Anaconda
 2.1 Follow the documentation: https://docs.anaconda.com/anaconda/install/windows/

3. Anaconda already installed a version of python, but you can have different version if you want. Then next step is to install virtualenv in order to control the version of python and libraries/packages version we might be needing during a project development. 
 3.1 to create a virtualenv just: 
 		conda create --name myenv python=3.5
 	in my case myenv is tesorflow, and python 3.5, but again you can use whatever version of python you need. 
 3.2. to activate the virtualenv after it is succefully install do: 
 		conda activate myenv
 3.3 Let's install packages to this virtualenv.
 	3.3.1 activate the virtualenv
	```bash
 		C:\Users\($yourpcname)>activate myenv
 		(myenv) C:\Users\($yourpcname)\pip install numpy
	```
 	3.3.2 the previous installation is an example to check if the virtualenv is working properly. now let's open python and import the just installed package: 
	```bash
 		(myenv) C:\Users\($yourpcname)>python
		Python 3.5.6 |Anaconda, Inc.| (default, Aug 26 2018, 16:05:27) [MSC v.1900 64 bit (AMD64)] on win32
		Type "help", "copyright", "credits" or "license" for more information.
		>>> import numpy
		>>> print (numpy.__version__)
		1.15.4
	```
	3.3.3 Now it is working very well. Nice let's go to the next step. 

4. So, we got sublime Text 3 in the step 1, we got anaconda and python in step 2, and we created a virtualenv in step 3. 
   	The step 4 is to write code in Sublime Text 3, and get the right enviroment to develop our projects. 
   	4.1 Creating the first project. I recomend the use of projects because it simplifies the custom configuration of the project. 
       Let's open Sublime Text 3, go to Project > Save project As , then put a name and save the file in a new folder. 
       For example I created myproject folder, and the project name is my_env_project. 
       In the folder you shall have two files: .sublime-project and .sublime-workspace extension files. 
   	4.2 We are going to go to Project > Edit Project and configure the virtualenv path of the python interpreter:
	```
		   {
			"build_systems":
			[
				{
					"file_regex": "^[ ]*File \"(...*?)\", line ([0-9]*)",
					"name": "Anaconda Python Builder",
					"selector": "source.python",
					"shell_cmd": "\"/Users/($yourpcname)/Anaconda3/envs/myenv/python.exe\" -u \"$file\""
				}
			],
			"folders":
			[
				{
					"follow_symlinks": true,
					"path": "."
				}
			],
			"settings":
			{
				"python_interpreter": "C:/Users/yourpcname/Anaconda3/envs/myenv/python.exe"
			}
		}
	 ```
	 
		//taken from : http://damnwidget.github.io/anaconda/anaconda_settings/
	4.3 The previous step helps to customize the python version of the virtualenv myenv, if you need to use another virtualenv
	I recomend you to create a new project and change the paths to your virtualenv paths. 

	4.4 Install Jedi, Anaconda and SublimeREPL to complete this. 
		4.4.1 Go to preferences > Package Control , then write install package. A new prompt window will appear. 
		4.4.2 Write Jedi, install it
		4.4.3 Repeat step 4.4.1 and then install Anaconda
		4.4.4 Repeat step 4.4.1 and then install SublimeREPL

	4.5 Configure Jedi, go to package settings > Jedi > Keymap - default. 
	    4.5.1 Look for the line: "command": "sublime_jedi_params_autocomplete", "keys": ["("]
	    and change it by "command": "sublime_jedi_params_autocomplete", "keys": ["ctrl+("],
	    it is because you won't be able to use the open parenthesis (, I am not sure why it is originally in that way. 
	    4.5.2 In the package settings > Jedi > Settings User:
	    ```
	    	{
				"settings":
				{
					"python_interpreter_path": "C:/Users/($yourpcname)/Anaconda3/envs/myenv/",
					"python_package_paths":
					[
						"C:/Users/($yourpcname)/Anaconda3/envs/Lib",
						"C:/Users/($yourpcname)/Anaconda3/envs/Lib/site-packages/"
					]
				}
			}
		```
		4.5.3 This is because we want Jedi to autocomplete the function name (members from a class), constant variables, etc. 

	4.6 Configure SublimeREPL, go to preferences > Browse Packages... , a explorer windows will show you the folder where 
	SublimeREPL is installed, but we need to go to SublimeREPL > config , and open the Python\Main.sublime-menu in sublime text 3 editor. 
		4.6.1 Add the following code and the end of the dictionary: 
		```
		,
                    {"command": "repl_open",
                     "caption": "Python MyEnv",
                     "id": "MyEnv env",
                     "mnemonic": "R",
                     "args": {
                        "type": "subprocess",
                        "encoding": "utf8",
                        "cmd": ["C:/Users/($yourpcname)/Anaconda3/envs/myenv/python", "-i", "$file_basename"],
                        "cwd": "$file_path",
                        "syntax": "Packages/Python/Python.tmLanguage",
                        "external_id": "python",
                        "extend_env": {"PYTHONIOENCODING": "utf-8"}
                        }
                    }
		 ```
        4.6.2 Save it, and we are almost done. 

    4.7 Configure the build system. Actually, it is select the generated Anaconda Python Builder. 
    If you go to step 4.2, we created a New Build System Specially for this project, no other project will have this Build system. 
    Look it is because we want an specific python interpreter for this project. Notice, we called -> name: "Anaconda Python Builder"
    Thus, go to tools > Build System , and click on Anaconda Python Builder. 

5. Let's try to run a conde in sublime text 3. 
	Add a new file in your project, but first go to view > View Side Bar. 
	You will see that you project has already a folder called with the name given at the moment to save the project name in step 4.1
	The new file will be saved in the same path.
	5.1 Lets go to file > new File. 
	5.2 Let's go to the bottom right corner of the sublime text 3 screen and where "Plain Text" is, please click on the text and a drop list will appear, then select python.
	5.3 automatically, the file unamed will understand your python syntaxys. Now, let's make some example code:
	```python
		import numpy
		print ( "hello words !!! ")
		print (numpy.__version__)
	```

	5.4 During the process of the code I hope you noticed how sublime text 3 is autocompleting your code by using the TAB after you write some words. 
	5.5 Now let's press Ctrl + B, to build the code or go to Tools > Build ... 
	5.6 The console will appear with the result of the code: 
	```	
	Hellow World !!!
	3.4.3
	[Finished in 1.0s]
 	```
 	5.7 Now, what if we want to run some code alive as Jupyter Notebook or sort of. 
 		Let's open SublimeREPL, go to Tools > SublimeREPL > python > Python MyEnv 
 		Notice, the name is the same as we created in step 4.6.1. Click on that and a new page/tab will appear. 
		

## License
[MIT](https://choosealicense.com/licenses/mit/)
