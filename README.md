# stable-diffusion-webui-deepdanbooru-tag2folder

Using this script you can move images using deepdanbooru classification

Version 1.1
* Add better detection of Anime and Character tags

Version 1.0
* Add initial commit and create repository

## Rules Explanation

The <Rules> section is a text area that will be filled using the rules to migrate the folder.
This extension reads the images in the <Source Folder> and then match the image with the rules.

The rules should be written in the following format:
{
	"subfolder1": {
		"AND": ["tag1", "tag2"],
	},
	"subfolder2": {
		"OR": ["tag1", "tag2"],
	}
}

Depending of the condition "AND" or "OR" will match the image to be moved in the subfolder.
Using "AND" will match all the tags in the list and using "OR" will match only one tag.
In case that an image does not match any of the rules, will be moved to a folder called "unclassified".

Additionally there are two more options in the <Automatic Type>, in case of the images does not belong to any rule. This options can create subfolders depending of the type
- None: will move the images to unclassified
- Anime: will fetch the anime serie that belong the image and create a new folder with that anime name
- Character: will fetch the character name that belong the image and create a new folder with that character name

You can review teh example input_folder and input_rules.json in the examples folder

![Interface](https://github.com/Jibaku789/sd-webui-deepdanbooru-tag2folder/blob/main/examples/interface.png)
