"""
URL configuration for music project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

from myapp import views

urlpatterns = [
    path('',views.login),
    path('home',views.home),
    path('user',views.user),
    path('user_search',views.user_search),
    path('song',views.song),
    path('song_search',views.song_search),
    path('book',views.book),
    path('book_search',views.book_search),
    path('rest',views.rest),
    path('rest_search',views.rest_search),
    path('feedback',views.feedback),
    path('feedback_search',views.feedback_search),
    path('addsong',views.addsong),
    path('addbook',views.addbook),
    path('addrest',views.addrest),
    path('loginPost',views.loginPost),
    path('addbookPost',views.addbookPost),
    path('addsongPost',views.addsongPost),
    path('addrestPost',views.addrestPost),
    path('deletebook/<id>',views.deletebook),
    path('editbook/<id>',views.editbook),
    path('editbookpost',views.editbookpost),
    path('deletesong/<id>',views.deletesong),
    path('editsong/<id>',views.editsong),
    path('editsongpost',views.editsongpost),
    path('deleterest/<id>',views.deleterest),
    path('editrest/<id>',views.editrest),
    path('editrestpost',views.editrestpost),

# -------------------android------------------------

    path('/viewsongs',views.viewsongs),
    path('/and_register',views.and_register),
    path('and_loginPost',views.and_loginPost),
    path('and_playlist',views.and_playlist),
    path('process_image',views.process_image),
    path('playlistemo',views.playlistemo),
    path('viewrest',views.viewrest),
    path('viewbook',views.viewbook),
    path('usersendfeedback',views.usersendfeedback),
    path('usersendrating',views.usersendrating),







]
