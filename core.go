package main

import (
	"html/template"
	"net/http"

	"github.com/gorilla/mux"
)

var tmpl = template.Must(template.ParseFiles("templates/dashboard.html"))

func main() {
	r := mux.NewRouter()
	r.HandleFunc("/", dashboardHandler)

	r.PathPrefix("/static/").Handler(http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))

	http.ListenAndServe(":8080", r)
}

// Dashboard handler
func dashboardHandler(w http.ResponseWriter, r *http.Request) {
	tmpl.Execute(w, nil)
}
