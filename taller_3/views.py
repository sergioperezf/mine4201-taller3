from django.shortcuts import render
from django.http import HttpResponse
from os.path import exists
from taller_3.services import get_possible_movies, recommend_service

def index(request):
	if (request.method == 'GET'):
		return render(request, 'taller_3/taller_3.html', {'movies': get_possible_movies()})
	else:
		movies = request.POST.getlist('movies')
		movies_int = []
		for movie in movies:
			movies_int.append(int(movie))
		recommendations = recommend_service(movies_int)
		return render(request, 'taller_3/taller_3.html', {'recommendations': recommendations, 'movies': get_possible_movies()})
