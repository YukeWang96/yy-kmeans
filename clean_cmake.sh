if [ -d CMakeFiles ]
then
	rm -rf CMakeFiles
	echo -e "CMakeFiles removed........\n"
fi

if [ -f CMakeCache.txt ]
then
	rm -r CMakeCache.txt
	echo -e "CMakeCache removed.........\n"
fi

if [ -f Makefile ]
then
	rm -r Makefile
	echo -e "Makefile removed...........\n"
fi

if [ ! -d CMakeFiles ] && [ ! -f CMakeCache.txt ] && [ ! -f Makefile ]
then
	echo -e "Clean All Done............\n"
fi
